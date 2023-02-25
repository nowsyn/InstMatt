import os
import glob
import logging
import functools
import numpy as np
import torch


class ImageFile(object):
    def __init__(self, phase='train'):
        self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names
        name_sets = [self._get_name_set(d) for d in dirs]

        # Reduce
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        if len(valid_names) == 0:
            self.logger.error('No image valid')
        else:
            self.logger.info('{}: {} foreground/images are valid'.format(self.phase.upper(), len(valid_names)))

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                 alpha_dir="train_alpha",
                 fg_dir="train_fg",
                 bg_dir="train_bg",
                 alpha_ext=".jpg",
                 fg_ext=".jpg",
                 bg_ext=".jpg"):
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir  = alpha_dir
        self.fg_dir     = fg_dir
        self.bg_dir     = bg_dir
        self.alpha_ext  = alpha_ext
        self.fg_ext     = fg_ext
        self.bg_ext     = bg_ext

        self.logger.debug('Load Training Images From Folders')

        self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir)
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)
        self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)

    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 alpha_ext=".png",
                 merged_ext=".png",
                 trimap_ext=".png"):
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext

        self.logger.debug('Load Testing Images From Folders')

        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)


class CustomImageFileTrain(object):
    def __init__(self, phase, alpha_dir, merged_dir, mask_dir, fg_dir, bg_dir, alpha_ext='.png', merged_ext='.jpg', mask_ext='.png'):
        self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.mask_dir = mask_dir
        self.fg_dir = fg_dir
        self.bg_dir = bg_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.mask_ext = mask_ext

        self.alpha = open(self.alpha_dir).read().splitlines()
        self.merged = open(self.merged_dir).read().splitlines()
        self.mask = open(self.mask_dir).read().splitlines()
        self.fg = open(self.fg_dir).read().splitlines()
        self.bg = open(self.bg_dir).read().splitlines()

        self._get_multiple_instances()

        assert len(self.alpha) == len(self.merged)
        assert len(self.alpha) == len(self.mask)
        assert len(self.alpha) == len(self.fg)
        assert len(self.alpha) == len(self.bg)


    def _get_multiple_instances(self):
        alpha_dict = {}
        mask_dict = {}
        fg_dict = {}
        merged_dict = {}
        bg_dict = {}

        for item in self.fg:
            path, name = os.path.split(item)
            if path not in fg_dict:
                fg_dict[path] = []
            fg_dict[path].append(item)

        for item in self.alpha:
            path, name = os.path.split(item)
            if path not in alpha_dict:
                alpha_dict[path] = []
            alpha_dict[path].append(item)

        for item in self.mask:
            path, name = os.path.split(item)
            if path not in mask_dict:
                mask_dict[path] = []
            mask_dict[path].append(item)

        for item in self.merged:
            splits = item.split('/')
            path = os.path.join('/'.join(splits[:-1]), splits[-1].split('.')[0])
            merged_dict[path] = item

        for item in self.bg:
            splits = item.split('/')
            path = os.path.join('/'.join(splits[:-1]), splits[-1].split('.')[0])
            bg_dict[path] = item

        merged_list = [merged_dict[key] for key in sorted(merged_dict.keys())]
        bg_list = [bg_dict[key] for key in sorted(bg_dict.keys())]
        alpha_list = [alpha_dict[key] for key in sorted(alpha_dict.keys())]
        mask_list = [mask_dict[key] for key in sorted(mask_dict.keys())]
        fg_list = [fg_dict[key] for key in sorted(fg_dict.keys())]

        print(len(merged_list), len(alpha_list), len(mask_list), len(fg_list), len(bg_list))

        assert len(merged_list) == len(alpha_list)
        assert len(merged_list) == len(mask_list)
        assert len(merged_list) == len(fg_list)
        assert len(merged_list) == len(bg_list)

        self.merged = merged_list
        self.alpha = alpha_list
        self.mask = mask_list
        self.fg = fg_list
        self.bg = bg_list


class CustomImageFileTest(object):
    def __init__(self, phase, alpha_dir, merged_dir, mask_dir, alpha_ext='.png', merged_ext='.jpg', mask_ext='.png'):
        self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.mask_dir = mask_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.mask_ext = mask_ext

        self.alpha = open(self.alpha_dir).read().splitlines()
        self.merged = open(self.merged_dir).read().splitlines()
        self.mask = open(self.mask_dir).read().splitlines()

        assert len(self.alpha) == len(self.merged)
        assert len(self.alpha) == len(self.mask)

        self._get_multiple_instances()

    def _get_multiple_instances(self):
        alpha_dict = {}
        mask_dict = {}

        for item in self.alpha:
            path, name = os.path.split(item)
            if path not in alpha_dict:
                alpha_dict[path] = []
            alpha_dict[path].append(item)

        for item in self.mask:
            path, name = os.path.split(item)
            if path not in mask_dict:
                mask_dict[path] = []
            mask_dict[path].append(item)

        merged_list = sorted(list(set(self.merged)))
        alpha_list = [alpha_dict[key] for key in sorted(alpha_dict.keys())]
        mask_list = [mask_dict[key] for key in sorted(mask_dict.keys())]
        assert len(merged_list) == len(alpha_list)
        assert len(merged_list) == len(mask_list)
        self.merged = merged_list
        self.alpha = alpha_list
        self.mask = mask_list
