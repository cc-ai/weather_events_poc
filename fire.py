import os
import torch
import numpy as np
import sys
import argparse
import json
import os
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.mixture import GaussianMixture


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--im_dir", type=str, required=True, help="Directory of images to transform",
)
parser.add_argument(
    "-m", "--mask_dir", type=str, required=True, help="Directory of sky masks",
)
parser.add_argument(
    "-br", "--blur_r", type=int, default=200, required=True, help="Blur radius",
)
parser.add_argument(
    "-r", "--red", type=int, default=255, required=True, help="Red intensity",
)
parser.add_argument(
    "-g", "--green", type=int, default=0, required=True, help="Green intensity",
)
parser.add_argument(
    "-b", "--blue", type=int, default=0, required=True, help="Blue intensity",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Directory where to save transformed images",
)
args = parser.parse_args()


def increase_sky_mask(sky_mask, increase=True, p_w=0, p_h=0):
    if p_h <= 0 and p_w <= 0:
        return sky_mask

    n_lines = int(p_h * sky_mask.shape[0])
    n_cols = int(p_w * sky_mask.shape[1])

    for i in range(1, n_cols):
        sky_mask[:, i::] += sky_mask[:, 0:-i]
        sky_mask[:, 0:-i] += sky_mask[:, i::]
    for i in range(1, n_lines):
        sky_mask[i::, :] += sky_mask[0:-i, :]
        sky_mask[0:-i, :] += sky_mask[i::, :]

    sky_mask[sky_mask >= 1] = 1

    return sky_mask


def add_fire(im, sky_mask, filter_color, blur_radius):
    # Darken the picture and increase contrast
    contraster = ImageEnhance.Contrast(im)
    im = contraster.enhance(2.0)
    darkener = ImageEnhance.Brightness(im)
    im = darkener.enhance(0.25)

    # Make the image more red
    im_array = np.array(im)
    im_array[:, :, 2] = np.minimum(im_array[:, :, 2], im_array[:, :, 2] - 20)
    im_array[:, :, 1] = np.minimum(im_array[:, :, 1], im_array[:, :, 1] - 10)
    im_array[:, :, 0] = np.maximum(im_array[:, :, 0], im_array[:, :, 0] + 40)
    im = Image.fromarray(im_array).convert("RGB")

    # Find sky proportion in picture
    num_sky_pixels = np.sum(sky_mask)
    sky_proportion = num_sky_pixels / (sky_mask.shape[0] * sky_mask.shape[1])
    has_sky = sky_proportion > 0.01

    # Adding red-ish color mostly in the sky
    if has_sky:
        filter_ = Image.new("RGB", im.size, filter_color)
        sky_mask = increase_sky_mask(sky_mask, True, 0.01, 0.01)
        im_mask = Image.fromarray((sky_mask * 255.0).squeeze()).convert("L")
        filter_mask = im_mask.filter(ImageFilter.GaussianBlur(blur_radius))
        im.paste(filter_, (0, 0), filter_mask)

    return im


if __name__ == "__main__":
    images_dir = args.im_dir
    sky_masks_dir = args.mask_dir
    save_dir = args.output
    blur_r = args.blur_r
    filter_color = (args.red, args.green, args.blue)

    ims_names = os.listdir(images_dir)
    masks_names = os.listdir(sky_masks_dir)

    ims_list = []
    masks_list = []
    im_names_with_mask = []

    print("\n- Creating lists of images and corresponding masks ...")
    for im_name in ims_names:
        for mask_name in masks_names:
            if os.path.splitext(im_name)[0] == os.path.splitext(mask_name)[0]:
                im = Image.open(os.path.join(images_dir, im_name)).convert("RGB")
                ims_list.append(im)
                mask = Image.open(os.path.join(sky_masks_dir, mask_name)).convert("1")
                mask = mask.resize(im.size, resample=Image.NEAREST)
                masks_list.append(np.array(mask))
                im_names_with_mask.append(os.path.splitext(im_name)[0])
    print("- Done.\n")

    for i, image in enumerate(ims_list):
        print("- Processing image", im_names_with_mask[i], "...")
        im_wildfire = add_fire(image, masks_list[i], filter_color, blur_r)
        im_wildfire.save(
            os.path.join(save_dir, im_names_with_mask[i] + ".png"), format="PNG"
        )
    print("- Done.\n")
