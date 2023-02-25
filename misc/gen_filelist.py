import os
import sys
import glob
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='masks/*.png')
    parser.add_argument('--image-prefix', type=str, default='images')
    parser.add_argument('--image-path', type=str, default='image.txt')
    parser.add_argument('--mask-path', type=str, default='mask.txt')

    args = parser.parse_args()

    mask_paths = sorted(glob.glob(os.path.join(args.input, "*/*.png")))

    t1 = open(args.image_path, "w")
    t2 = open(args.mask_path, "w")

    for mask_path in mask_paths:
        splits = mask_path.split('/')
        img_path = os.path.join(args.image_prefix, splits[-2]+'.jpg')
        assert os.path.exists(img_path)
        t1.write(img_path+'\n')
        t2.write(mask_path+'\n')
