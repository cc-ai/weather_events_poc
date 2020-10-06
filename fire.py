from PIL import Image, ImageEnhance
import torch
import numpy as np
import sys
import json
import os
from sklearn.mixture import GaussianMixture


def normalize(arr, mini=0, maxi=1):
    # breakpoint()
    return mini + (maxi - mini) * ((arr - arr.min()) / (arr.max() - arr.min()))


def normalize_depth(arr, mini=0, maxi=1):
    # breakpoint()
    return mini + (maxi - mini) * arr / arr.max()


def normalize_depth_2(arr, arr_min, arr_max, mini=0, maxi=1):
    # breakpoint()
    return mini + (maxi - mini) * (arr - arr_min) / (arr_max - arr_min)

def get_sky_cutoff(depth, thresh = 80):
    """
    suppose that depth distribution of pixels with value < 80 follows gaussian mixture distribution, the one with lowest mean value corresponding to sky area
    depth is np.array of inverse depth values normalized between 0 and 255
    """  
    mixture = GaussianMixture(n_components=2).fit(depth[depth<thresh].reshape(-1, 1))
    means_hat = mixture.means_.flatten()
    sds_hat = np.sqrt(mixture.covariances_).flatten()
    ind_min = np.argmin(means_hat)
    ind_max = np.argmax(means_hat)
    print(means_hat[ind_max] - means_hat[ind_min])
    if (means_hat[ind_max] - means_hat[ind_min])>25:
        cutoff = min(means_hat[ind_min] + 2*sds_hat[ind_min], means_hat[ind_max] - 2*sds_hat[ind_max])
    else:
        cutoff = 0
    return(cutoff)
    


# 219,1,1
def add_fire(
    im,
    depth_array,
    use_blend=False,
    blending_alpha=0.2,
    filter_color=(235, 111, 50),
    other_color=(2, 42, 42),
    blending_color=(219, 1, 1),
    resize=None,
):
    # Darkening the picture
    enhancer = ImageEnhance.Brightness(im)
    im = enhancer.enhance(0.3)

    # Warming the picture
    im_array = np.array(im)
    im_array[:, :, 2] = np.minimum(im_array[:, :, 2], im_array[:, :, 2] - 20)
    im_array[:, :, 1] = np.minimum(im_array[:, :, 1], im_array[:, :, 1] - 10)
    im_array[:, :, 0] = np.maximum(im_array[:, :, 0], im_array[:, :, 0] + 40)
    im = Image.fromarray(im_array).convert("RGBA")

    # Adding bright red/orange mostly in the sky, scaled with depth
    #threshold = 300
    #if depth_array.min() < threshold:  # may be sky in the picture
    depth_array = (255 * normalize(depth_array)).astype(np.uint8)
    # Adding bright red/orange mostly in the sky, scaled with depth
        
    cutoff = get_sky_cutoff(depth_array, thresh = 80)
    if cutoff > 0:
        inds = (depth_array<cutoff)
        depth_array[(1-inds).astype(bool)] = normalize(
                depth_array[(1-inds).astype(bool)],
                max(cutoff, depth_array.min()),
                depth_array.max(),
            )
        min_norm = 0.25
        depth = normalize_depth_2(depth_array, 0.0, depth_array.max(), min_norm, 1.0)
        depth = 1 / depth
        depth = normalize_depth_2(depth, 1.0, 1.0 / min_norm)

        im_depth = Image.fromarray((depth * 255.0).squeeze()).convert("L")

        if resize is None:  # Resize depth to im size
            im_depth = im_depth.resize(im.size)
        else:
            im.thumbnail(resize, Image.ANTIALIAS)
            im_depth.thumbnail(resize, Image.ANTIALIAS)

        depth = np.array(im_depth)

        filter_ = Image.new("RGB", np.transpose(depth).shape, filter_color)

        im.paste(filter_, (0, 0), im_depth)

    if use_blend:
        smogged = Image.blend(
            im, Image.new("RGBA", im.size, color=blending_color), alpha=blending_alpha
        )
    else:
        smogged = im

    return smogged


if __name__ == "__main__":
    path_to_json = sys.argv[1]
    save_dir = sys.argv[2]

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
        im_path = elem["x"]
        print(im_path)
        im = Image.open(im_path).convert("RGBA")
        input_im = transform(np.array(im)[:, :, :3]).to(device)

        # Infer depth map with MiDaS
        with torch.no_grad():
            depth_array = midas(input_im).cpu().numpy()

        im_smogged = add_fire(im, depth_array, filter_color=(219,1,1))
        im_smogged.save(
            os.path.join(save_dir, os.path.basename(elem["x"])), format="PNG"
        )
