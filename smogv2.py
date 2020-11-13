import comet_ml
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import noise
import sys
import argparse
from pathlib import Path


def get_sky_mask(sky_path):
    seg_tens = torch.load(sky_path).squeeze()
    sky_mask = seg_tens == 9
    sky_mask = F.interpolate(
        sky_mask.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor),
        (im.size[1], im.size[0]),
    )
    sky_mask = sky_mask.squeeze().cpu().detach().numpy().astype(bool)
    return sky_mask


def srgb2lrgb(I0):
    I0 = normalize(I0)
    I = ((I0 + 0.055) / 1.055) ** (2.4)
    I[I0 <= 0.04045] = I0[I0 <= 0.04045] / 12.92
    return I


def lrgb2srgb(I1):
    I2 = np.zeros(I1.shape)
    for k in range(3):
        temp = I1[:, :, k]

        I2[:, :, k] = 12.92 * temp * (temp <= 0.0031308) + (
            1.055 * np.power(temp, (1 / 2.4)) - 0.055
        ) * (temp > 0.0031308)
    return I2


def normalize(arr, arr_min=None, arr_max=None, mini=0, maxi=1):
    if arr_min is None:
        arr_min = arr.min()
    if arr_max is None:
        arr_max = arr.max()
    return mini + (maxi - mini) * (arr - arr_min) / (arr_max - arr_min)


def add_smog(
    img_path,
    depth_array,
    sky_path=None,
    pert_perlin=False,
    airlight=0.76,
    beta_param=2,
    vr=1,
    exp=None,
):

    A = airlight * np.ones(3)

    im = Image.open(img_path)
    I0 = np.array(im)[:, :, :3]
    I = srgb2lrgb(I0)

    # depth_array= normalize(np.array(Image.open(depth_path)))*255
    min_norm = 0.1
    depth = normalize(depth_array, arr_min=None, arr_max=None, mini=0.3, maxi=1.0)
    depth = 1.0 / depth
    depth = normalize(depth, arr_min=None, arr_max=None, mini=0.1, maxi=1)
    depth = np.array(Image.fromarray(depth).resize(im.size))
    if not sky_path is None:
        sky_mask = get_sky_mask(sky_path)
        depth[sky_mask] = 1
    im_depth = Image.fromarray((depth * 255.0).squeeze()).convert("L")
    im_depth = im_depth.resize(im.size)
    d = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
    # Add perlin noise for visual vividness(see reference 16 in the HazeRD paper). The noise is added to the depth
    # map, which is equivalent to change the optic distance.
    if pert_perlin:
        d = d * ((perlin_noise(np.zeros(d.shape)) - 0.5) + 1)
        d = normalize(d)
    # convert depth map to transmission
    beta = np.array([beta_param / vr, beta_param / vr, beta_param / vr])
    transmission = np.exp(d * -beta)

    # Obtain simulated linear RGB hazy image. Eq. 3 in the HazeRD paper
    Ic = np.multiply((transmission), I) + np.multiply(A, 1 - transmission)

    # convert linear RGB to sRGB
    I2 = lrgb2srgb(Ic)

    if exp is not None:
        exp.log_image(I2, img_path)
        exp.log_parameter("perlin_noise", pert_perlin)
        exp.log_parameter("beta", beta_param)
        exp.log_parameter("airlight", airlight)

    return I2


def perlin_noise(world, scale=600.0, octaves=6, persistence=0.5, lacunarity=2.0):
    shape = world.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=shape[0],
                repeaty=shape[1],
                base=0,
            )
    return world


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", "-j", type=str)
    parser.add_argument("--save_dir", "-s", type=str, default="")
    parser.add_argument("--comet", "-c", action="store_true", default=False)
    parser.add_argument("--no_pert", action="store_true", default=False)
    args = parser.parse_args()

    path_to_json = Path(args.json)

    assert path_to_json.exists()

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
        assert parser.comet, "Specify a --save_dir or use --comet"

    use_perlin_perturbation = not args.no_pert

    exp = None
    if args.comet:
        exp = comet_ml.Experiment(project_name="smogv2")

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

    for elem in data:
        im_path = elem["x"]
        print(im_path)
        im = Image.open(im_path).convert("RGBA")
        input_im = transform(np.array(im)[:, :, :3]).to(device)

        # Infer depth map with MiDaS
        with torch.no_grad():
            depth_array = midas(input_im).squeeze().cpu().numpy()

        I2 = add_smog(
            im_path,
            depth_array,
            sky_path=None,
            pert_perlin=use_perlin_perturbation,
            exp=exp,
        )
        if save_dir is not None:
            im_smogged = Image.fromarray((255 * normalize(I2)).astype(np.uint8))
            im_smogged.save(
                os.path.join(save_dir, os.path.basename(elem["x"])), format="PNG"
            )
