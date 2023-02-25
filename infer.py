import os
import cv2
import toml
import argparse
import numpy as np
import time
import tqdm

import torch
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks

from scipy.ndimage import morphology

def dilate(mask, t=20):
    unknown = morphology.distance_transform_edt(mask==0) <= t
    return unknown


def transform(image_dict, alpha_pred, thumbnail=False):
    if not thumbnail:
        h, w = image_dict['alpha_shape']
        alpha_pred = alpha_pred[:, 0, ...].data.cpu().numpy()
        alpha_pred = np.transpose(alpha_pred, (1,2,0))
        alpha_pred = alpha_pred * 255
        alpha_pred = alpha_pred.astype(np.uint8)
        alpha_pred = alpha_pred[32:h+32, 32:w+32]
        if alpha_pred.shape[0] != image_dict['oh'] or alpha_pred.shape[1] != image_dict['ow']:
            alpha_pred = cv2.resize(alpha_pred, (image_dict['ow'], image_dict['oh']))
    else:
        n, c, _, _ = alpha_pred.shape
        h, w = image_dict['alpha_shape']
        alpha_pred = alpha_pred.data.cpu().numpy()
        alpha_pred = alpha_pred[:,:,32:h+32,32:w+32]
        alpha_pred = alpha_pred.transpose((0,2,1,3))
        alpha_pred = alpha_pred.reshape(n,h,w*c)
        alpha_pred = np.transpose(alpha_pred, (1,2,0))
        alpha_pred = alpha_pred * 255
        alpha_pred = alpha_pred.astype(np.uint8)
    if len(alpha_pred.shape) == 2:
        alpha_pred = alpha_pred[:,:,None]
    return alpha_pred


def single_inference(model, image_dict, thumbnail=False):
    model.cuda()

    with torch.no_grad():
        image, mask = image_dict['image'], image_dict['mask']
        alpha_shape = image_dict['alpha_shape']
        image = image.cuda()
        mask = mask.cuda()

        pred = [model(image[i:i+1], mask[i:i+1], is_training=False) for i in range(mask.size(0))]
        pred = utils.reduce_dict(pred)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        weight_os4 = weight_os4.max(dim=1, keepdim=True)[0]
        alpha_pred = alpha_pred * (weight_os4<=0).float() + alpha_pred_os4 * (weight_os4>0).float()
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        weight_os1 = weight_os1.max(dim=1, keepdim=True)[0]
        alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_os1 * (weight_os1>0).float()

        if model.refiner is not None:
            alpha_pred_list = model.forward_refiner(image, alpha_pred.clone().detach(), pred['feature'].clone().detach(), is_training=False, nostop=False)
            alpha_pred = alpha_pred * (weight_os1<=0).float() + alpha_pred_list[-1] * (weight_os1>0).float()

    return transform(image_dict, alpha_pred, thumbnail=thumbnail)


def generator_tensor_dict(image_path, mask_paths, max_side=1920):
    # read images
    image = cv2.imread(image_path)
    mask = np.stack([(cv2.imread(mask_path, 0).astype(np.float32)/255.) for mask_path in mask_paths], axis=2) # h*w*k
    oh, ow = image.shape[:2]
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if max(ow, oh)>max_side:
        ratio = max_side / max(ow, oh)
        nh = int(ratio * oh)
        nw = int(ratio * ow)
        image = cv2.resize(image, (nw, nh))
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if len(mask.shape) == 2:
        mask = mask[:,:,None]

    sample = {'image': image, 'mask': mask, 'alpha_shape': mask.shape[:2]}

    # reshape
    h, w = sample["alpha_shape"]

    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,32), (32, 32), (0,0)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # convert GBR images to RGB
    image, mask = sample['image'][:,:,::-1], sample['mask']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)
    mask = mask.transpose((2, 0, 1)).astype(np.float32)

    # normalize image
    image /= 255.

    sample['image'] = torch.from_numpy(image)
    sample['image'] = sample['image'].sub_(mean).div_(std)
    sample['mask'] = torch.from_numpy(mask)
    mask_batch = []
    for i in range(sample['mask'].shape[0]):
        mask_t = sample['mask'][i]
        mask_r = torch.sum(sample['mask'], dim=0) - mask_t
        mask_b = 1 - torch.sum(sample['mask'], dim=0)
        mask = torch.stack([mask_t, mask_r, mask_b], dim=0)
        mask_batch.append(mask)
    mask_batch = torch.stack(mask_batch, dim=0)
    sample['mask'] = mask_batch

    # add first channel
    sample['image'] = sample['image'][None, ...].repeat(sample['mask'].shape[0],1,1,1)

    sample['ow'] = ow
    sample['oh'] = oh

    return sample


def get_multiple_instances(image_list, mask_list):
    mask_dict = {}
    for item in mask_list:
        path, name = os.path.split(item)
        if path not in mask_dict:
            mask_dict[path] = []
        mask_dict[path].append(item)
    image_list = sorted(list(set(image_list)))
    mask_list = [mask_dict[key] for key in sorted(mask_dict.keys())]
    assert len(image_list) == len(mask_list)
    return image_list, mask_list


if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/InstMatt.toml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/InstMatt/best_model.pth', help="path of checkpoint")
    parser.add_argument('--image-dir', type=str, default='datasets/images', help="input image dir")
    parser.add_argument('--mask-dir', type=str, default='datasets/masks', help="input mask dir")
    parser.add_argument('--image-ext', type=str, default='.png', help="input image ext")
    parser.add_argument('--mask-ext', type=str, default='.png', help="input mask ext")
    parser.add_argument('--output', type=str, default='results', help="output dir")
    parser.add_argument('--guidance-thres', type=int, default=128, help="guidance input threshold")
    parser.add_argument('--max-side', type=int, default=1280, help="maximum side length")

    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    args.output = os.path.join(args.output, CONFIG.version+'_'+args.checkpoint.split('/')[-1])
    utils.make_dir(args.output)

    # build model
    model = networks.get_generator(
        CONFIG,
        encoder=CONFIG.model.arch.encoder,
        decoder=CONFIG.model.arch.decoder,
        refiner=CONFIG.model.arch.refiner,
    )
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()

    image_list = open(args.image_dir).read().splitlines()
    mask_list = open(args.mask_dir).read().splitlines()
    image_list, mask_list = get_multiple_instances(image_list, mask_list)

    os.makedirs(args.output, exist_ok=True)

    for idx, (image_path, mask_paths) in enumerate(zip(image_list, mask_list)):

        print(f"Processing [{idx+1}/{len(mask_list)}] ...")

        image_name = os.path.basename(image_path).split('.')[0]
        os.makedirs(os.path.join(args.output, image_name), exist_ok=True)

        image_dict = generator_tensor_dict(image_path, mask_paths, max_side=args.max_side)

        alpha_pred = single_inference(model, image_dict, thumbnail=False)

        for i in range(alpha_pred.shape[2]):
            cv2.imwrite(os.path.join(args.output, image_name, os.path.basename(mask_paths[i])), alpha_pred[:,:,i])
