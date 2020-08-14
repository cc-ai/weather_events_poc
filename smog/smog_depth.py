from PIL import Image
import numpy as np
import sys
import json
import os

def normalize(arr):
    return((arr - arr.min())/(arr.max() - arr.min()))

def add_smog(im_path, depth_path, filter_color = (177, 177, 179,255), resize=None):
    im = Image.open(im_path).convert('RGBA')
    depth = np.exp(np.load(depth_path))
    im_depth = Image.fromarray(normalize(depth)*255).convert('L')
    if resize is None: #resize depth  to im size
        im_depth= im_depth.resize(im.size)
    else: 
        im.thumbnail(resize, Image.ANTIALIAS)
        im_depth.thumbnail(resize, Image.ANTIALIAS)
    depth = np.array(im_depth)
    depth = normalize(depth)
    filter_ = Image.new('L', np.transpose(depth).shape, 255)
    filter_= np.dstack((np.array(filter_), depth*255)).astype(np.uint8)

    filter_ = Image.fromarray(filter_).convert('RGBA')
    im.paste(filter_, (0, 0), filter_)
    smogged = Image.blend(im, Image.new('RGBA', im.size, color = filter_color), alpha = 0.5)
    return(smogged)

if __name__=="__main__":

    path_to_json = sys.argv[1]
    save_dir = sys.argv[2]
    with open(path_to_json, 'r') as f: 
        data = json.load(f)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for elem in data:
        im= add_smog(elem['x'], elem['d'])
        im.save(os.path.join(save_dir, os.path.basename(elem['x'])))