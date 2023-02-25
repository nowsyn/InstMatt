import cv2
import os
import math
import numbers
import random
import logging
import numpy as np
import imgaug.augmenters as iaa

import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms

from   utils import CONFIG

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test", real_world_aug = False):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase
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
        image, alpha, mask, cmask = sample['image'][:,:,::-1], sample['alpha'], sample['mask'], sample['cmask']
        if 'trimap' in sample:
            trimap = sample['trimap']
        else:
            trimap = None
        
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
     
        if self.phase == 'train' and self.RWA is not None and np.random.rand() < 0.5:
            image[image > 255] = 255
            image[image < 0] = 0
            image = np.round(image).astype(np.uint8)
            image = np.expand_dims(image, axis=0)
            image = self.RWA(images=image)
            image = image[0, ...]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        # alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        # mask = np.expand_dims(mask.astype(np.float32), axis=0)
        if trimap is not None:
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 2
            trimap[trimap >= 85] = 1

        # normalize image
        sample['image'] = torch.from_numpy(image/255.).unsqueeze(0)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['alpha'] = torch.from_numpy(alpha).unsqueeze(1).float()
        sample['mask'] = torch.from_numpy(mask).unsqueeze(1).float()
        sample['cmask'] = torch.from_numpy(cmask).unsqueeze(1).float()

        if trimap is not None:
            # trimap: kxhxw or kxcxhxw
            sample['trimap'] = torch.from_numpy(trimap).to(torch.long)
            if CONFIG.model.trimap_channel == 3:
                sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(0,3,1,2).float()
            elif CONFIG.model.trimap_channel == 1:
                sample['trimap'] = sample['trimap'].unsqueeze(1).float()
            else:
                raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")
        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
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

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
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

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        image, alpha, mask, cmask = sample['image'], sample['alpha'], sample['mask'], sample['cmask']
        rows, cols, ch = image.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, image.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, image.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        image = cv2.warpAffine(image, M, (cols, rows), flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = np.stack(
            [cv2.warpAffine(item, M, (cols, rows), flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP) for item in alpha],
            axis=0)
        mask = np.stack(
            [cv2.warpAffine(item, M, (cols, rows), flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP) for item in mask],
            axis=0
        )
        cmask = np.stack(
            [cv2.warpAffine(item, M, (cols, rows), flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP) for item in cmask],
            axis=0
        )

        sample['image'], sample['alpha'], sample['mask'], sample['cmask'] = image, alpha, mask, cmask

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

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        image, alpha, trimap, mask, cmask, name = \
            sample['image'],  sample['alpha'], sample['trimap'], sample['mask'], sample['cmask'], sample['image_name']

        pick = random.choice(list(range(trimap.shape[0])))
        h, w = trimap[pick].shape
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                image = cv2.resize(image, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = np.stack(
                    [cv2.resize(item, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST)) for item in alpha],
                    axis=0)
                trimap = np.stack(
                    [cv2.resize(item, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST) for item in trimap],
                    axis=0)
                mask = np.stack(
                    [cv2.resize(item, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST) for item in mask],
                    axis=0)
                cmask = np.stack(
                    [cv2.resize(item, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST) for item in cmask],
                    axis=0)
                h, w = trimap[pick].shape
        small_trimap = cv2.resize(trimap[pick], (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        image_crop = image[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[:,left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        trimap_crop = trimap[:,left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        mask_crop = mask[:,left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        cmask_crop = cmask[:,left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap[pick]==128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            image_crop = cv2.resize(image, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = np.stack(
                [cv2.resize(item, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST)) for item in alpha],
                axis=0)
            trimap_crop = np.stack(
                [cv2.resize(item, self.output_size[::-1], interpolation=cv2.INTER_NEAREST) for item in trimap],
                axis=0)
            mask_crop = np.stack(
                [cv2.resize(item, self.output_size[::-1], interpolation=cv2.INTER_NEAREST) for item in mask],
                axis=0)
            cmask_crop = np.stack(
                [cv2.resize(item, self.output_size[::-1], interpolation=cv2.INTER_NEAREST) for item in cmask],
                axis=0)
        
        sample.update({'image': image_crop, 'alpha': alpha_crop, 'trimap': trimap_crop, 'mask': mask_crop, 'cmask': cmask_crop})
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
        padded_mask = np.pad(sample['mask'], ((0,0), (0,pad_h), (0, pad_w)), mode="reflect")
        padded_cmask = np.pad(sample['cmask'], ((0,0), (0,pad_h), (0, pad_w)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask
        sample['cmask'] = padded_cmask

        if 'trimap' in sample:
            padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")
            sample['trimap'] = padded_trimap
        return sample


class GenMask(object):
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
        trimap = np.stack([self.generate(item) for item in sample['alpha']], axis=0)
        sample['trimap'] = trimap
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
    def __init__(self, perturb_prob = 0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            mask = sample['mask'] # H x W, trimap 0--255, segmask 0--1, alpha 0--1
            h, w = mask.shape
            perturb_size_h, perturb_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
            x = random.randint(0, h - perturb_size_h)
            y = random.randint(0, w - perturb_size_w)
            x1 = random.randint(0, h - perturb_size_h)
            y1 = random.randint(0, w - perturb_size_w)
            mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1+perturb_size_h, y1:y1+perturb_size_w].copy()
            sample['mask'] = mask

        return sample

class DataGenerator(Dataset):
    def __init__(self, data, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size
        self.alpha = data.alpha
        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.merged = []
            self.trimap = []
        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap

        train_trans = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            CutMask(perturb_prob=CONFIG.data.cutmask_prob),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug) ]

        test_trans = [ OriginScale(), ToTensor() ]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)

    def __getitem__(self, idx):
        if self.phase == "train":
            fg = cv2.imread(self.fg[idx % self.fg_num])
            alpha = cv2.imread(self.alpha[idx % self.fg_num], 0).astype(np.float32)/255
            bg = cv2.imread(self.bg[idx], 1)

            fg, alpha = self._composite_fg(fg, alpha, idx)

            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name}

        else:
            image = cv2.imread(self.merged[idx])
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            trimap = cv2.imread(self.trimap[idx], 0)
            mask = (trimap >= 170).astype(np.float32)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape}

        sample = self.transform(sample)

        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if  np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def __len__(self):
        if self.phase == "train":
            return len(self.bg)
        else:
            return len(self.alpha)


class CustomDataGenerator(Dataset):
    def __init__(self, data, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size
        self.i = 0
        if self.phase == "train":
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.mask = data.mask
            self.alpha = data.alpha
        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged[:100]
            self.mask = data.mask[:100]
            self.alpha = data.alpha[:100]

        train_trans = [
            RandomAffine(degrees=5, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            RandomCrop((self.crop_size, self.crop_size)),
            # RandomJitter(),
            ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug) ]

        test_trans = [OriginScale(), ToTensor()]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)

    def calc_iou(self, m1, m2):
        union = np.logical_or(m1[:,None,:,:], m2[None,:,:,:]).sum(axis=(2,3))
        inter = np.logical_and(m1[:,None,:,:], m2[None,:,:,:]).sum(axis=(2,3))
        return inter / (union + 1e-8)
    
    def assign_label(self, alpha, mask):
        iou_matrix = self.calc_iou(alpha, mask)
        iou = np.max(iou_matrix, axis=1)
        iou_idx = np.argmax(iou_matrix, axis=1)
        iou_weight = iou>0.5
        keep_alpha = alpha[np.where(iou>0.5)]
        keep_mask = mask[iou_idx][np.where(iou>0.5)]
        if len(keep_mask)==0:
            keep_alpha = alpha
            keep_mask = mask[iou_idx]
        assert keep_alpha.shape == keep_mask.shape 
        return keep_alpha, keep_mask

    def combine_mask(self, alpha, mask):
        k = mask.shape[0]
        pick1 = random.sample(list(range(k)), k=min(k,random.choice(list(range(1,4)))))
        ret_alpha = alpha[pick1] 
        ret_mask = mask[pick1]
        ret_cmask = np.max(mask, axis=0, keepdims=True) - ret_mask

        pick2 = list(set(list(range(k))) - set(pick1))
        if len(pick2)>0:
            # pick2 = random.sample(list(range(k)), k=min(k,2))
            comp_alpha = np.sum(alpha[pick2], axis=0, keepdims=True)
            comp_mask = np.max(mask[pick2], axis=0, keepdims=True)
            comp_cmask = np.max(mask,axis=0,keepdims=True) - comp_mask 
            ret_alpha = np.concatenate([ret_alpha, comp_alpha], axis=0)
            ret_mask = np.concatenate([ret_mask, comp_mask], axis=0)
            ret_cmask = np.concatenate([ret_cmask, comp_cmask], axis=0)
        return ret_alpha, ret_mask, ret_cmask

    def __getitem__(self, idx):
        image = cv2.imread(self.merged[idx])
        alpha_list = np.stack([cv2.imread(item, 0)/255. for item in self.alpha[idx]], axis=0)
        mask_list = np.stack([cv2.imread(item, 0)/255. for item in self.mask[idx]], axis=0)
        alpha, mask = self.assign_label(alpha_list, mask_list)
        if self.phase == "train":
            alpha, mask, cmask = self.combine_mask(alpha, mask)
        else:
            cmask = np.max(mask,axis=0,keepdims=True) - mask
        # image: 3xhxw, alpha: kxhxw, mask: kxhxw, cmask: kxhxw
        image_name = os.path.split(self.merged[idx])[-1]

        sample = {
            'image': image, 
            'alpha': alpha, 
            'mask': mask, 
            'cmask': cmask,
            'image_name': image_name, 
            'alpha_shape': alpha.shape[-2:]
        }

        sample = self.transform(sample)
        # if self.phase == "train":
        #     rows = []
        #     for key in ['alpha', 'mask', 'cmask']:
        #         data = sample[key][:,0].data.cpu().numpy() * 255
        #         data = np.concatenate([np.sum(data, axis=0, keepdims=True), data], axis=0)
        #         rows.append(data)
        #     rows = np.stack(rows, axis=0)
        #     m, n, h, w = rows.shape
        #     rows = np.reshape(np.transpose(rows, (1,2,0,3)), (n*h,m*w))
        #     rows = cv2.resize(rows, None, fx=0.5, fy=0.5)
        #     cv2.imwrite('debug/%04d.png' % (self.i,), rows)
        
        self.i += 1
        return sample

    def __len__(self):
        return len(self.alpha)


def batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    # image: k*3*h*w --> maxk*3*h*w
    # mask: m*h*w --> maxm*1*h*w
    collated_batch = {} 
    max_length = max([sample['mask'].shape[0] for sample in batch])
    h, w = batch[0]['image'].size(2), batch[0]['image'].size(3)
    for key in batch[0].keys():
        if key in ['image']: 
            collated_batch[key] = torch.cat([sample[key].repeat(max_length, 1, 1, 1) for sample in batch], dim=0)
        elif key in ['alpha', 'mask', 'cmask', 'trimap']:
            data = batch[0][key]
            template = torch.zeros((len(batch), max_length, data.size(1), data.size(2), data.size(3))).float()
            for i, sample in enumerate(batch):
                data = sample[key]
                template[i, 0:data.size(0)] = data
            collated_batch[key] = template.view(-1, data.size(1), data.size(2), data.size(3)) 
        else:
            collated_batch[key] = [sample[key] for sample in batch]
    return collated_batch
