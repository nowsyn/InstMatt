import os
import cv2
import torch
import logging
import numpy as np
from utils.config import CONFIG
import torch.distributed as dist
import torch.nn.functional as F
from skimage.measure import label
from collections import OrderedDict

def make_dir(target_dir):
    """
    Create dir if not exists
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def print_network(model, name):
    """
    Print out the network information
    """
    logger = logging.getLogger("Logger")
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()

    logger.info(model)
    logger.info(name)
    logger.info("Number of parameters: {}".format(num_params))


def update_lr(lr, optimizer):
    """
    update learning rates
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr(init_lr, step, iter_num):
    """
    Warm up learning rate
    """
    return step/iter_num*init_lr


def add_prefix_state_dict(state_dict, prefix="module"):
    """
    add prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[prefix+"."+key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    return new_state_dict


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict


def load_imagenet_pretrain(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location = lambda storage, loc: storage.cuda(CONFIG.gpu))
    state_dict = remove_prefix_state_dict(checkpoint['state_dict'])
    for key, value in state_dict.items():
        state_dict[key] = state_dict[key].float()

    logger = logging.getLogger("Logger")
    logger.debug("Imagenet pretrained keys:")
    logger.debug(state_dict.keys())
    logger.debug("Generator keys:")
    logger.debug(model.module.encoder.state_dict().keys())
    logger.debug("Intersection  keys:")
    logger.debug(set(model.module.encoder.state_dict().keys())&set(state_dict.keys()))

    weight_u = state_dict["conv1.module.weight_u"]
    weight_v = state_dict["conv1.module.weight_v"]
    weight_bar = state_dict["conv1.module.weight_bar"]

    logger.debug("weight_v: {}".format(weight_v))
    logger.debug("weight_bar: {}".format(weight_bar.view(32, -1)))
    logger.debug("sigma: {}".format(weight_u.dot(weight_bar.view(32, -1).mv(weight_v))))

    new_weight_v = torch.zeros((3+CONFIG.model.mask_channel), 3, 3).cuda()
    new_weight_bar = torch.zeros(32, (3+CONFIG.model.mask_channel), 3, 3).cuda()

    new_weight_v[:3, :, :].copy_(weight_v.view(3, 3, 3))
    new_weight_bar[:, :3, :, :].copy_(weight_bar)

    logger.debug("new weight_v: {}".format(new_weight_v.view(-1)))
    logger.debug("new weight_bar: {}".format(new_weight_bar.view(32, -1)))
    logger.debug("new sigma: {}".format(weight_u.dot(new_weight_bar.view(32, -1).mv(new_weight_v.view(-1)))))

    state_dict["conv1.module.weight_v"] = new_weight_v.view(-1)
    state_dict["conv1.module.weight_bar"] = new_weight_bar

    model.module.encoder.load_state_dict(state_dict, strict=False)


def load_VGG_pretrain(model, checkpoint_file):
    """
    Load imagenet pretrained resnet
    Add zeros channel to the first convolution layer
    Since we have the spectral normalization, we need to do a little more
    """
    checkpoint = torch.load(checkpoint_file, map_location = lambda storage, loc: storage.cuda())
    backbone_state_dict = remove_prefix_state_dict(checkpoint['state_dict'])

    model.module.encoder.load_state_dict(backbone_state_dict, strict=False)


def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if CONFIG.model.trimap_channel == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(1).float()
    return weight


def get_gaborfilter(angles):
    """
    generate gabor filter as the conv kernel
    :param angles: number of different angles
    """
    gabor_filter = []
    for angle in range(angles):
        gabor_filter.append(cv2.getGaborKernel(ksize=(5,5), sigma=0.5, theta=angle*np.pi/8, lambd=5, gamma=0.5))
    gabor_filter = np.array(gabor_filter)
    gabor_filter = np.expand_dims(gabor_filter, axis=1)
    return gabor_filter.astype(np.float32)


def get_gradfilter():
    """
    generate gradient filter as the conv kernel
    """
    grad_filter = []
    grad_filter.append([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_filter.append([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    grad_filter = np.array(grad_filter)
    grad_filter = np.expand_dims(grad_filter, axis=1)
    return grad_filter.astype(np.float32)


def reduce_tensor_dict(tensor_dict, mode='mean'):
    """
    average tensor dict over different GPUs
    """
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            tensor_dict[key] = reduce_tensor(tensor, mode)
    return tensor_dict


def reduce_tensor(tensor, mode='mean'):
    """
    average tensor over different GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= CONFIG.world_size
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt

### preprocess the image and mask for inference (np array), crop based on ROI
def preprocess(image, mask, thres):
    mask_ = (mask >= thres).astype(np.float32)
    arr = np.nonzero(mask_)
    h, w = mask.shape
    bbox = [max(0, int(min(arr[0]) - 0.1*h)),
            min(h, int(max(arr[0]) + 0.1*h)),
            max(0, int(min(arr[1]) - 0.1*w)),
            min(w, int(max(arr[1]) + 0.1*w))]
    image = image[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return image, mask, bbox

### postprocess the alpha prediction to keep the largest connected component (np array) and uncrop, alpha in [0, 1]
### based on https://github.com/senguptaumd/Background-Matting/blob/master/test_background-matting_image.py
def postprocess(alpha, orih=None, oriw=None, bbox=None):
    labels=label((alpha>0.05).astype(int))
    try:
        assert( labels.max() != 0 )
    except:
        return None
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    alpha = alpha * largestCC
    if bbox is None:
        return alpha
    else:
        ori_alpha = np.zeros(shape=[orih, oriw], dtype=np.float32)
        ori_alpha[bbox[0]:bbox[1], bbox[2]:bbox[3]] = alpha
        return ori_alpha


Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,50)]
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape

    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    uncertain_area[pred>1-1.0/255.0] = 0

    for n in range(N):
        for c in range(C):
            uncertain_area_ = uncertain_area[n,c,:,:] # H, W
            if train_mode:
                width = np.random.randint(1, rand_width)
            else:
                width = rand_width // 2
            uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
            uncertain_area[n,c,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).cuda()

    return weight


def get_solid_tensor_from_alpha(alpha, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = alpha.shape

    alpha = alpha.data.cpu().numpy()
    boundary_area = np.ones_like(alpha, dtype=np.uint8)
    fg = (alpha == 1)
    bg = (alpha == 0)

    for n in range(N):
        for c in range(C):
            fg_area_ = fg[n,c,:,:] # H, W
            bg_area_ = bg[n,c,:,:] # H, W
            ratio = (fg_area_.sum() + bg_area_.sum()) / float(H*W)
            # print(ratio, (alpha[n,c]==0).sum()+(alpha[n,c]==1).sum(), H*W)
            if ratio >= 0.99:
                width = 35
                fg_area_ = cv2.dilate(fg_area_.astype(np.uint8), Kernels[width])
                bg_area_ = cv2.dilate(bg_area_.astype(np.uint8), Kernels[width])
                boundary_area[n,c,:,:] = 0
                boundary_area[n,c,fg_area_==0] = 1
                boundary_area[n,c,bg_area_==0] = 1

    # debug = np.reshape(np.transpose(boundary_area, (0,2,1,3)), (N*H, C*W)) * 255
    # cv2.imwrite("debug.jpg", debug)
    weight = torch.from_numpy(boundary_area).float().cuda()
    return weight


def filter_mismatch_keys(state_dict1, state_dict2):
    keep_state_dict = OrderedDict() 
    for key in state_dict1:
        if key in state_dict2:
            if state_dict1[key].shape == state_dict2[key].shape:
                keep_state_dict[key] = state_dict2[key]
    print('{} keys are removed.'.format(len(state_dict2.keys())-len(keep_state_dict.keys())))
    return keep_state_dict


def reduce_dict(list_of_dict):
    reduced_dict = {}
    for key in list_of_dict[0].keys():
        reduced_dict[key] = torch.cat([sample[key] for sample in list_of_dict], dim=0)
    return reduced_dict


def reduce_list(list_of_list):
    reduced_list = []
    for i in range(len(list_of_list[0])):
        reduced_list.append(torch.cat([sample[i] for sample in list_of_list], dim=0))
    return reduced_list


def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def group_reduce_sum(tensor, bs):
    n, c, h, w = tensor.shape
    return tensor.view(bs,n//bs,c,h,w).sum(dim=1, keepdim=True).repeat(1,n//bs,1,1,1).view(n,c,h,w)


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
