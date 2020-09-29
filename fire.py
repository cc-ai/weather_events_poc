from PIL import Image, ImageEnhance
import torch
import numpy as np
import sys
import json
import os


def normalize(arr, mini=0, maxi=1):
    return mini + (maxi - mini) * ((arr - arr.min()) / (arr.max() - arr.min()))


# 181,84,55
def add_fire(
    im,
    depth_array,
    use_blend=False,
    blending_alpha=0.2,
    filter_color=(219, 1, 1),
    other_color=(255, 139, 50),
    blending_color=(219, 1, 1),
    resize=None,
):
    # Warming the picture
    im_array = np.array(im)
    im_array[:, :, 2] = np.minimum(im_array[:, :, 2], im_array[:, :, 2] - 20)
    im_array[:, :, 1] = np.minimum(im_array[:, :, 1], im_array[:, :, 1] - 10)
    im_array[:, :, 0] = np.maximum(im_array[:, :, 0], im_array[:, :, 0] + 40)
    im = Image.fromarray(im_array).convert("RGBA")

    # Darkening the picture
    enhancer = ImageEnhance.Brightness(im)
    im = enhancer.enhance(0.25)

    # Adding bright red/orange mostly in the sky, scaled with depth
    depth = normalize(depth_array, 0.3, 1.0)
    depth = normalize(1.0 / depth)
    treshold = 0.85
    depth[depth < treshold] = normalize(depth[depth < treshold] ** 3, 0, treshold)

    im_depth = Image.fromarray((depth * 200.0).squeeze()).convert("L")

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
        im = Image.open(im_path).convert("RGBA")
        input_im = transform(np.array(im)[:, :, :3]).to(device)

        # Infer depth map with MiDaS
        with torch.no_grad():
            depth_array = midas(input_im).cpu().numpy()

        im_smogged = add_fire(im, depth_array)
        im_smogged.save(
            os.path.join(save_dir, os.path.basename(elem["x"])), format="PNG"
        )
