from PIL import Image
import torch
import numpy as np
import sys
import json
import os


def normalize(arr, mini=0, maxi=1):
    return mini + (maxi - mini) * ((arr - arr.min()) / (arr.max() - arr.min()))


def add_smog(
    im,
    depth_array,
    use_blend=True,
    blending_alpha=0.5,
    filter_color=(255, 255, 255),
    blending_color=(105, 99, 88, 255),
    resize=None,
):
    depth = normalize(depth_array, 0.2, 1.0)
    depth = normalize(1.0 / depth)
    im_depth = Image.fromarray((depth * 255.0).squeeze()).convert("L")

    if resize is None:  # Resize depth to im size
        im_depth = im_depth.resize(im.size)
    else:
        im.thumbnail(resize, Image.ANTIALIAS)
        im_depth.thumbnail(resize, Image.ANTIALIAS)

    depth = np.array(im_depth)
    depth = normalize(depth, 0, 255)

    filter_ = Image.new("RGB", np.transpose(depth).shape, filter_color)

    filter_ = np.dstack((np.array(filter_), depth)).astype(np.uint8)
    filter_ = Image.fromarray(filter_).convert("RGBA")
    im.paste(filter_, (0, 0), filter_)

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

        im_smogged = add_smog(im, depth_array)
        im_smogged.save(
            os.path.join(save_dir, os.path.basename(elem["x"])), format="PNG"
        )
