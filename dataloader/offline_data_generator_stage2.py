import cv2
import os
import math
import numbers
import random
import logging
import numpy as np
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms

from utils import CONFIG, normalize_image

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]

def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

class ToTensorTrain(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, real_world_aug = False):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        if real_world_aug:
            self.RWA = iaa.SomeOf((1, None), [
                iaa.LinearContrast((0.6, 1.4)),
                iaa.JpegCompression(compression=(0, 60)),
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
            ], random_order=True)
        else:
            self.RWA = None

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, mask = sample['image'][:,:,::-1], sample['alpha'], sample['mask']

        if self.RWA is not None and np.random.rand() < 0.5:
            image[image > 255] = 255
            image[image < 0] = 0
            image = np.round(image).astype(np.uint8)
            image = self.RWA(images=image)

        # normalize image
        h, w, k = sample['mask'].shape
        assert k % 3 == 0
        k = k // 3

        image = image.transpose((2,0,1))
        sample['image'] = torch.from_numpy(image/255.).float()
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['image'] = sample['image'].unsqueeze(0).repeat(k,1,1,1)
        sample['alpha'] = torch.from_numpy(np.clip(alpha.transpose((2,0,1)),0,1)).float()
        sample['alpha'] = sample['alpha'].view(k,3,h,w)
        sample['mask'] = torch.from_numpy(np.clip(mask.transpose((2,0,1)),0,1)).float()
        sample['mask'] = sample['mask'].view(k,3,h,w)

        h, w = image.shape[1:3]
        fg = sample['fg'].reshape((h,w,k*3,3))[...,::-1].transpose((2,3,0,1))
        sample['fg'] = torch.from_numpy(fg/255.).float()
        sample['fg'] = sample['fg'].sub_(self.mean.view(1,3,1,1)).div_(self.std.view(1,3,1,1))
        sample['fg'] = sample['fg'].view(k,3,3,h,w)

        trimap = sample['trimap']
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1

        sample['trimap'] = torch.from_numpy(trimap).to(torch.long)
        if CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'].permute(2,0,1).float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 1")
        sample['trimap'] = sample['trimap'].view(k,3,h,w)
        return sample


class ToTensorTest(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, real_world_aug = False):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, mask = sample['image'][:,:,::-1], sample['alpha'], sample['mask']
        h, w = image.shape[:2]

        # normalize image
        image = image.transpose((2,0,1))
        sample['image'] = torch.from_numpy(image/255.).float()
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['image'] = sample['image'].unsqueeze(0).repeat(alpha.shape[0],1,1,1)
        sample['alpha'] = torch.from_numpy(np.clip(alpha,0,1)).unsqueeze(1).float()
        sample['mask'] = torch.from_numpy(np.clip(mask.transpose(0,3,1,2),0,1)).float()
        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, vertical_flip=True, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip
        self.vertical_flip = vertical_flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size, vertical_flip):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1
            # disable vertical flip
            if not vertical_flip:
                flip[1] = 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        image, fg, alpha, mask = sample['image'], sample['fg'], sample['alpha'], sample['mask']
        rows, cols, ch = image.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, image.size, self.vertical_flip)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, image.size, self.vertical_flip)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        image = cv2.warpAffine(image, M, (cols, rows), flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        fg = cv2.warpAffine(fg, M, (cols, rows), flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows), flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        mask = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

        sample['image'], sample['fg'], sample['alpha'], sample['mask'] = image, fg, alpha, mask

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['image'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['image'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'
    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        image, fg, alpha, trimap, mask, name = sample['image'],  sample['fg'], sample['alpha'], sample['trimap'], sample['mask'], sample['image_name']

        h, w = trimap[...,0].shape[:2]
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                image = cv2.resize(image, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize(mask, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap[...,0].shape[:2]

        small_trimap = cv2.resize(trimap[...,0], (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4, self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        image_crop = image[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        mask_crop = mask[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap==128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            image_crop = cv2.resize(image, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)

        sample.update({'image': image_crop, 'fg': fg_crop, 'alpha': alpha_crop, 'trimap': trimap_crop, 'mask': mask_crop})
        return sample


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((0,0), (0,pad_h), (0,pad_w), (0,0)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask

        if 'trimap' in sample:
            padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0,pad_w), (0,0)), mode="reflect")
            sample['trimap'] = padded_trimap
        return sample


class RescaleTest(object):
    def __init__(self, min_scale=1000, max_scale=1400):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # alpha, mask, fg: khw
        # image, bg: hw
        h, w = image.shape[:2]
        if min(h,w)<self.min_scale:
            ratio = self.min_scale / float(min(h,w))
            h,w = int(h*ratio), int(w*ratio)
        if max(h,w)>self.max_scale:
            ratio = self.max_scale / float(max(h,w))
            h,w = int(h*ratio), int(w*ratio)

        mask = np.stack([cv2.resize(mask_i, (w,h), interpolation=cv2.INTER_NEAREST) for mask_i in mask], axis=0)
        image = cv2.resize(image, (w,h))

        sample['rescaled_alpha_shape'] = (h,w)

        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(image, ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_mask = np.pad(mask, ((0,0), (0,pad_h), (0, pad_w), (0,0)), mode="reflect")

        sample['image'] = padded_image
        sample['mask'] = padded_mask
        return sample


class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    def generate(self, alpha):
        h, w = alpha.shape

        max_kernel_size = max(30, int((min(h,w) / 2048) * 30))

        ### generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        return trimap

    def __call__(self, sample):
        sample['trimap'] = np.stack([self.generate(sample['alpha'][:,:,i]) for i in range(sample['alpha'].shape[2])], axis=2)
        return sample


class AssignMask(object):
    def __init__(self, phase, threshold=0):
        self.phase = phase
        self.threshold = threshold

    def calc_iou(self, m1, m2):
        union = np.logical_or(m1[:,None,:,:], m2[None,:,:,:]).sum(axis=(2,3))
        inter = np.logical_and(m1[:,None,:,:], m2[None,:,:,:]).sum(axis=(2,3))
        return inter / (union + 1e-8)

    def __call__(self, sample):
        iou_matrix = self.calc_iou(sample['alpha'], sample['mask'])
        iou_max = np.max(iou_matrix, axis=1)
        iou_idx = np.argmax(iou_matrix, axis=1)
        sample['mask'], sample['iou'] = sample['mask'][iou_idx], iou_max
        if self.phase == "test":
            sample['mask'] = sample['mask'][iou_max>self.threshold]
            sample['alpha'] = sample['alpha'][iou_max>self.threshold]
        return sample


class GenMaskTrain(object):
    def __init__(self, random_sampling=0.25):

        self.random_sampling = random_sampling
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    def generate(self, alpha):
        h, w = alpha.shape
        max_kernel_size = max(30, int((min(h,w) / 2048) * 30))
        thres = random.uniform(0.01, 1.0)
        seg_mask = (alpha >= thres).astype(np.int).astype(np.uint8)
        random_num = random.randint(0,3)

        if random_num == 0:
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 1:
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 2:
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 3:
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        return seg_mask.astype(np.float32)

    def combine(self, fg, alpha, mask, pick1=None):
        n_list = list(range(0, mask.shape[0]))
        if pick1 is None:
            pick1 = random.sample(n_list, k=random.randint(0, mask.shape[0]-1))
        pick2 = list(set(n_list) - set(pick1))

        if len(pick1) == 0:
            alpha_t = np.zeros((fg.shape[1], fg.shape[2]))
            mask_t = np.zeros((fg.shape[1], fg.shape[2]))
            fg_t = np.zeros((fg.shape[1], fg.shape[2], 3))
        else:
            alpha_t = np.sum(alpha[pick1], axis=0)
            mask_t = np.max(mask[pick1], axis=0)
            fg_t = np.sum(alpha[pick1][...,None]*fg[pick1], axis=0) / (alpha_t[...,None]+1e-8)

        if len(pick2) == 0:
            alpha_r = np.zeros((fg.shape[1], fg.shape[2]))
            mask_r = np.zeros((fg.shape[1], fg.shape[2]))
            fg_r = np.zeros((fg.shape[1], fg.shape[2], 3))
        else:
            alpha_r = np.sum(alpha[pick2], axis=0)
            mask_r = np.max(mask[pick2], axis=0)
            fg_r = np.sum(alpha[pick2][...,None]*fg[pick2], axis=0) / (alpha_r[...,None]+1e-8)

        if len(pick2)>1 and random.random() < self.random_sampling:
            pick3 = random.sample(pick2, k=len(pick2)-1)
            mask_r = np.max(mask[pick3], axis=0)

        fg = np.concatenate([fg_t, fg_r], axis=2)
        alpha = np.stack([alpha_t, alpha_r], axis=2)
        mask = np.stack([mask_t, mask_r], axis=2)

        return fg, alpha, mask

    def __call__(self, sample):
        # generate mask from gt
        auto_mask = np.stack([self.generate(item) for item in sample['alpha']], axis=0)
        condition = (sample['iou']>random.uniform(0.3,0.5)).reshape((-1,1,1))
        sample['mask'] = np.where(condition, sample['mask'], auto_mask)

        # generate mask, t, r, b
        alpha_bg = 1 - np.sum(sample['alpha'], axis=0)
        mask_bg = 1 - np.sum(sample['mask'], axis=0)

        fg_list, alpha_list, mask_list = [], [], []

        k = sample['mask'].shape[0]
        order = random.sample(list(range(k)), k=k)
        for i in order:
            fg, alpha, mask = self.combine(sample['fg'], sample['alpha'], sample['mask'], pick1=[i])
            fg = np.concatenate([fg, sample['bg']], axis=2) # hw9
            alpha = np.concatenate([alpha, alpha_bg[...,None]], axis=2) # hw3
            mask = np.concatenate([mask, mask_bg[...,None]], axis=2) # hw3
            mask = np.stack([self.generate(mask[...,i]) for i in range(mask.shape[2])], axis=2)
            fg_list.append(fg)
            alpha_list.append(alpha)
            mask_list.append(mask)

        sample['fg'] = np.concatenate(fg_list, axis=2)
        sample['alpha'] = np.concatenate(alpha_list, axis=2)
        sample['mask'] = np.concatenate(mask_list, axis=2)

        del sample['iou']
        del sample['bg']

        return sample


class GenMaskTest(object):
    def __init__(self):
        pass

    def combine(self, mask, i):
        pick1 = [i]
        pick2 = list(set(list(range(mask.shape[0]))) - set(pick1))
        mask_t = np.max(mask[pick1], axis=0)
        mask_r = np.max(mask[pick2], axis=0)
        mask = np.stack([mask_t, mask_r], axis=2)
        return mask

    def __call__(self, sample):
        mask_bg = 1 - np.sum(sample['mask'], axis=0)
        mask_batch = []
        for i in range(sample['mask'].shape[0]):
            mask = self.combine(sample['mask'], i)
            mask_batch.append(np.concatenate([mask, mask_bg[...,None]], axis=2))
        sample['mask'] = np.stack(mask_batch, axis=0)
        del sample['iou']
        return sample


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class CutMask(object):
    def __init__(self, internal_perturb_prob = 0, external_perturb_prob = 0):
        self.internal_perturb_prob = internal_perturb_prob
        self.external_perturb_prob = external_perturb_prob

    def internal(self, mask):
        if np.random.rand() < self.internal_perturb_prob:
            h, w = mask.shape
            perturb_size_h, perturb_size_w = random.randint(h // 8, h // 4), random.randint(w // 8, w // 4)
            x = random.randint(0, h - perturb_size_h)
            y = random.randint(0, w - perturb_size_w)
            x1 = random.randint(0, h - perturb_size_h)
            y1 = random.randint(0, w - perturb_size_w)
            mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1+perturb_size_h, y1:y1+perturb_size_w].copy()
        return mask

    def external(self, mask):
        if np.random.rand() < self.external_perturb_prob:
            i, j = random.sample(list(range(0, mask.shape[2])), k=2)
            h, w = mask.shape[:2]
            perturb_size_h, perturb_size_w = random.randint(h // 8, h // 4), random.randint(w // 8, w // 4)
            x = random.randint(0, h - perturb_size_h)
            y = random.randint(0, w - perturb_size_w)
            x1 = random.randint(0, h - perturb_size_h)
            y1 = random.randint(0, w - perturb_size_w)
            mask_i_perturb = mask[x:x+perturb_size_h, y:y+perturb_size_w, i].copy()
            mask_j_perturb = mask[x:x+perturb_size_h, y:y+perturb_size_w, j].copy()
            mask[x:x+perturb_size_h, y:y+perturb_size_w, i] = mask_j_perturb
            mask[x:x+perturb_size_h, y:y+perturb_size_w, j] = mask_i_perturb
        return mask

    def __call__(self, sample):
        if random.random() < 0.5:
            sample['mask'] = np.stack([self.internal(sample['mask'][...,i]) for i in range(sample['mask'].shape[2])], axis=2)
        else:
            sample['mask'] = self.external(sample['mask'])
        return sample


class CustomDataGenerator(Dataset):
    def __init__(self, data, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size

        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.merged = data.merged
            self.mask = data.mask
            self.alpha = data.alpha
        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.mask = data.mask
            self.alpha = data.alpha

        train_trans = [
            AssignMask(phase="train"),
            GenMaskTrain(random_sampling=CONFIG.data.mask_random_sampling),
            RandomAffine(degrees=5, scale=[0.8, 1.25], shear=10, flip=0.5, vertical_flip=CONFIG.data.vertical_flip),
            GenTrimap(),
            RandomCrop((self.crop_size, self.crop_size)),
            CutMask(internal_perturb_prob=CONFIG.data.cutmask_internal_prob, external_perturb_prob=CONFIG.data.cutmask_external_prob),
            ToTensorTrain(real_world_aug=CONFIG.data.real_world_aug)
        ]

        test_trans = [
            AssignMask(phase="test", threshold=CONFIG.test.mask_threshold),
            GenMaskTest(),
            RescaleTest(max_scale=CONFIG.test.max_scale),
            ToTensorTest()
        ]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose(test_trans),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)

    def __getitem__(self, idx):
        image = cv2.imread(self.merged[idx])
        alpha = np.stack([cv2.imread(item,0)/255. for item in self.alpha[idx]], axis=0) # khw
        mask = np.stack([cv2.imread(item,0)/255. for item in self.mask[idx]], axis=0) # khw
        image_name = os.path.split(self.merged[idx])[-1]

        if self.phase == "train":
            bg = cv2.imread(self.bg[idx])
            fg = np.stack([cv2.imread(item) for item in self.fg[idx]], axis=0) # khwc
            assert fg.shape[0] == alpha.shape[0]
            sample = {
                'alpha': alpha, # khw
                'image': image, # 1hwc
                'mask': mask, # khw
                'fg': fg,
                'bg': bg,
                'image_name': image_name,
                'alpha_shape': alpha.shape[1:3]
            }
        else:
            sample = {
                'alpha': alpha, # khw
                'image': image, # 1hwc
                'mask': mask, # khw
                'image_name': image_name,
                'alpha_shape': alpha.shape[1:3]
            }

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.alpha)


def batch_collator(batch):
    collated_batch = {}
    h, w = batch[0]['image'].size(2), batch[0]['image'].size(3)
    for key in batch[0].keys():
        if key in ['image', 'fg', 'alpha', 'mask', 'trimap']:
            collated_batch[key] = torch.cat([sample[key] for sample in batch], dim=0)
        else:
            collated_batch[key] = [sample[key] for sample in batch]
    return collated_batch
