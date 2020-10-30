from PIL import Image, ImageEnhance, ImageFilter
import torch
import numpy as np
import sys
import json
import os
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def normalize(arr, arr_min=None, arr_max=None, mini=0, maxi=1):
    if arr_min is None:
        arr_min = arr.min()
    if arr_max is None:
        arr_max = arr.max()
    return mini + (maxi - mini) * (arr - arr_min) / (arr_max - arr_min)


def get_sky_cutoff(depth, thresh=80):
    """
    suppose that depth distribution of pixels with value < 80 follows gaussian mixture distribution, the one with lowest mean value corresponding to sky area
    depth is np.array of inverse depth values normalized between 0 and 255
    """
    mixture = GaussianMixture(n_components=2).fit(depth[depth < thresh].reshape(-1, 1))
    means_hat = mixture.means_.flatten()
    sds_hat = np.sqrt(mixture.covariances_).flatten()
    ind_min = np.argmin(means_hat)
    ind_max = np.argmax(means_hat)

    if (means_hat[ind_max] - means_hat[ind_min]) > 25:
        cutoff = min(
            means_hat[ind_min] + sds_hat[ind_min],
            means_hat[ind_max] - sds_hat[ind_max],
        )
    else:
        cutoff = 0
    return cutoff


def get_cutoff(v, intervals, num_sky_pixels):
    v[0] = 0  # Maximum will be at 0, we want to find the other max
    max = v.argmax()
    for i in range(max, v.shape[0]):
        if v[i] / num_sky_pixels < 0.01:
            return intervals[i]
    return None


def change_sky_zone(sky_mask, increase=True, p_w=0, p_h=0):
    if p_h <= 0 and p_w <= 0:
        return sky_mask

    n_lines = int(p_h * sky_mask.shape[0])
    n_cols = int(p_w * sky_mask.shape[1])

    new_mask = np.copy(sky_mask)

    if increase:
        for i in range(1, n_cols):
            new_mask[:, i::] += sky_mask[:, 0:-i]
            new_mask[:, 0:-i] += sky_mask[:, i::]
        for i in range(1, n_lines):
            new_mask[i::, :] += new_mask[0:-i, :]
            new_mask[0:-i, :] += new_mask[i::, :]
    else:
        for i in range(1, n_cols):
            new_mask[:, i::] &= sky_mask[:, 0:-i]
            new_mask[:, 0:-i] &= sky_mask[:, i::]
        for i in range(1, n_lines):
            new_mask[i::, :] &= new_mask[0:-i, :]
            new_mask[0:-i, :] &= new_mask[i::, :]

    if increase:
        new_mask[new_mask >= 1] = 1
    # else:
    #     maxi = np.max(new_mask)
    #     new_mask[new_mask < maxi] = 0

    return new_mask


def add_fire(
    im,
    depth_array,
    sky_mask,
    entropy,
    use_seg=True,
    use_depth=False,
    use_blend=False,
    degrade_colors=False,
    filter_color=(235, 36, 15),
    blending_color=(219, 1, 1),
):
    if degrade_colors and use_seg:
        color_1 = (255, 148, 0)
        color_2 = (255, 148, 0)
    else:
        color_1 = (255, 0, 0)

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
    im = Image.fromarray(im_array).convert("RGBA")

    # Find sky proportion in picture
    num_sky_pixels = np.sum(sky_mask)
    sky_proportion = num_sky_pixels / (sky_mask.shape[0] * sky_mask.shape[1])
    has_sky = sky_proportion > 0.01

    # Adding bright red/orange mostly in the sky
    if has_sky:
        filter_1 = Image.new("RGB", im.size, color_1)
        if degrade_colors and use_seg:
            filter_2 = Image.new("RGB", im.size, color_2)

        if use_depth:
            # Get sky cutoff
            sky_depth = sky_mask * depth_array
            v, intervals = np.histogram(sky_depth.ravel(), bins=50)
            cutoff = get_cutoff(v, intervals, num_sky_pixels)

            if cutoff is not None:
                # Push all depth values in the foreground to the maximum
                depth_array[depth_array > cutoff] = normalize(
                    np.log(depth_array[depth_array > cutoff]), cutoff, 255.0
                )

                # Compute the inverse of the (inverse depth)
                min_norm = 0.3
                depth = normalize(depth_array, min_norm, 1.0)
                depth = 1.0 / depth
                depth = normalize(depth)

                im_depth = Image.fromarray((depth * 255.0).squeeze()).convert("L")
                im_depth = im_depth.resize(im.size)

                # Paste on picture
                im.paste(filter_1, (0, 0), im_depth)

        if use_seg:
            # Compute first mask with small Gaussian blur and paste on picture
            sky_mask1 = change_sky_zone(sky_mask, True, 0.01, 0.01)
            # sky_mask1 = sky_mask
            im_mask = Image.fromarray((sky_mask1 * 200.0).squeeze()).convert("L")
            blur_radius = 200
            # blur_radius = normalize(entropy, 0, 3.46, 5, 500)
            # if blur_radius > 10:
            #     sky_mask1 = change_sky_zone(sky_mask, True, 0.01, 0.01)
            mask1 = im_mask.filter(ImageFilter.GaussianBlur(blur_radius))
            im.paste(filter_1, (0, 0), mask1)  # mask1

            if degrade_colors:
                # Compute second mask with bigger Gaussian blur
                sky_mask2 = change_sky_zone(sky_mask, False, 0.02, 0.02)
                mask2_ = Image.fromarray((sky_mask2 * 220.0).squeeze()).convert("L")
                mask2_ = mask2_.filter(ImageFilter.GaussianBlur(100))  # 300 works well
                mask2 = Image.fromarray((sky_mask * 0.0).squeeze()).convert("L")
                mask2.paste(mask2_, (0, 0), mask1)
                im.paste(filter_2, (0, 0), mask2_)

        darkener = ImageEnhance.Brightness(im)
        im = darkener.enhance(0.9)

    if use_blend:
        smogged = Image.blend(
            im, Image.new("RGBA", im.size, color=blending_color), alpha=0.25
        )
    else:
        smogged = im

    return smogged


if __name__ == "__main__":
    path_to_json = sys.argv[1]
    save_dir = sys.argv[2]
    use_seg = int(sys.argv[3])
    use_depth = int(sys.argv[4])
    use_blend = int(sys.argv[5])
    degrade_colors = int(sys.argv[6])

    if use_depth:
        # Load MiDaS model
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        midas.to(device)
        midas.eval()

        # Load MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.default_transform

    with open(path_to_json, "r") as f:
        data = json.load(f)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for elem in data:
        # Load input image
        im_path = elem["x"]
        print(im_path)
        im = Image.open(im_path).convert("RGBA")
        depth_array = None

        # Load seg map and retrieve sky_mask
        # Normally, here, would infer the seg_map with infer.py
        seg_tens = torch.load(elem["s"]).squeeze()
        seg_ind = torch.argmax(seg_tens, dim=0)
        sky_mask = seg_ind == 9
        breakpoint()
        seg_tens[:, seg_ind] = 0  # DEXTER
        seg_ind2 = torch.argmax(seg_tens, dim=0)  # DEXTER
        sky_mask2 = seg_ind2 == 9  # DEXTER
        global_mask = sky_mask + sky_mask2  # DEXTER

        seg_ent = torch.load(elem["e"])
        entropy1 = torch.sum(seg_ent[global_mask]) / torch.sum(global_mask)
        entropy = torch.sum(seg_ent[sky_mask]) / torch.sum(sky_mask)
        print(entropy1)
        print(entropy)

        sky_mask = F.interpolate(
            sky_mask.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor),
            (im.size[1], im.size[0]),
        )
        sky_mask = sky_mask.squeeze().cpu().detach().numpy().astype(bool)

        if use_depth:
            input_im = transform(np.array(im)[:, :, :3]).to(device)
            # Infer depth map with MiDaS
            with torch.no_grad():
                depth = midas(input_im).cpu().numpy()

            # Resize depth
            im_depth = Image.fromarray(normalize(depth.squeeze()) * 255).convert("L")
            im_depth = im_depth.resize(im.size, resample=Image.NEAREST)
            depth_array = np.array(im_depth)

        im_smogged = add_fire(
            im,
            depth_array,
            sky_mask,
            entropy,
            use_seg,
            use_depth,
            use_blend,
            degrade_colors,
        )
        im_smogged.save(
            os.path.join(save_dir, os.path.basename(elem["x"])), format="PNG"
        )
