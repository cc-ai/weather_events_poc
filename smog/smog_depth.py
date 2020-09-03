from PIL import Image
import numpy as np
import sys
import json
import os


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def add_smog(
    im_path,
    depth_path,
    seg_path,
    use_seg,
    use_blend,
    blending_alpha,
    filter_color=(213, 162, 20),
    blending_color=(200, 200, 200, 255),
    resize=None,
):
    im = Image.open(im_path).convert("RGBA")
    im_seg = Image.open(seg_path).convert("RGBA")
    depth = np.load(depth_path)
    im_depth = Image.fromarray(normalize(depth) * 255).convert("L")

    if resize is None:  # resize depth to im size
        im_depth = im_depth.resize(im.size)
        im_seg = im_seg.resize(im.size)
    else:
        im.thumbnail(resize, Image.ANTIALIAS)
        im_seg.thumbnail(resize, Image.ANTIALIAS)
        im_depth.thumbnail(resize, Image.ANTIALIAS)

    depth = np.array(im_depth)
    depth = normalize(depth)

    filter_ = Image.new("RGB", np.transpose(depth).shape, filter_color)

    if use_seg:
        seg = np.array(im_seg)

        # Define the filters for sky zone and no-sky zone
        sky_filter = np.dstack((np.array(filter_), 100 * np.ones(depth.shape))).astype(
            np.uint8
        )
        no_sky_filter = np.dstack((np.array(filter_), depth * 200)).astype(np.uint8)

        # Consider depth fot the no-sky zone
        # Sky is class [8, 19, 49, 255], so we can only look at R value to know if it's sky
        no_sky_mask = (
            seg[:, :, 0] != (8 * np.ones((seg.shape[0], seg.shape[1])))
        ).astype(np.uint8)
        no_sky_filter[:, :, 3] = no_sky_mask * no_sky_filter[:, :, 3]
        no_sky_filter = Image.fromarray(no_sky_filter).convert("RGBA")
        im.paste(no_sky_filter, (0, 0), no_sky_filter)

        # Uniform pasting for the sky-zone
        sky_mask = (seg[:, :, 0] == (8 * np.ones((seg.shape[0], seg.shape[1])))).astype(
            np.uint8
        )
        sky_filter[:, :, 3] = sky_mask * sky_filter[:, :, 3]
        sky_filter = Image.fromarray(sky_filter).convert("RGBA")
        im.paste(sky_filter, (0, 0), sky_filter)
    else:
        filter_ = np.dstack((np.array(filter_), depth * 128)).astype(np.uint8)
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
    with open(path_to_json, "r") as f:
        data = json.load(f)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for elem in data:
        im = add_smog(
            im_path=elem["x"],
            depth_path=elem["d"],
            seg_path=elem["s"],
            use_seg=False,
            use_blend=True,
            blending_alpha=0.5,
        )
        im.save(os.path.join(save_dir, os.path.basename(elem["x"])), format="PNG")

