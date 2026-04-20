"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def open_image(path, size):
    img = Image.open(path).convert('RGB')
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def load_segment(path, size):
    if not os.path.exists(path):
        return None
    segment = Image.open(path).convert('L')
    if size:
        segment = segment.resize((size, size), Image.NEAREST)
    segment = np.array(segment)
    return segment.astype(np.int)


def save_image(img, path):
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)


def change_seg(seg):
    color_dict = {
        (0, 0, 255): 3,  # blue
        (0, 255, 0): 2,  # green
        (0, 0, 0): 0,  # black
        (255, 255, 255): 1,  # white
        (255, 0, 0): 4,  # red
        (255, 255, 0): 5,  # yellow
        (128, 128, 128): 6,  # grey
        (0, 255, 255): 7,  # lightblue
        (255, 0, 255): 8  # purple
    }
    arr_seg = np.asarray(seg)
    new_seg = np.zeros(arr_seg.shape[:-1])
    for x in range(arr_seg.shape[0]):
        for y in range(arr_seg.shape[1]):
            if tuple(arr_seg[x, y, :]) in color_dict:
                new_seg[x, y] = color_dict[tuple(arr_seg[x, y, :])]
            else:
                min_dist_index = 0
                min_dist = 99999
                for key in color_dict:
                    dist = np.sum(np.abs(np.asarray(key) - arr_seg[x, y, :]))
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = color_dict[key]
                    elif dist == min_dist:
                        try:
                            min_dist_index = new_seg[x, y-1, :]
                        except Exception:
                            pass
                new_seg[x, y] = min_dist_index
    return new_seg.astype(np.uint8)


def compute_label_info(content_segment, style_segment):
    # Handle None segments (when no segmentation masks are provided)
    if content_segment is None or style_segment is None:
        return None, None
    # Handle empty arrays
    if not isinstance(content_segment, np.ndarray) or not isinstance(style_segment, np.ndarray):
        return None, None
    if content_segment.size == 0 or style_segment.size == 0:
        return None, None
    max_label = np.max(content_segment) + 1
    label_set = np.unique(content_segment)
    label_indicator = np.zeros(max_label)
    for label in label_set:
        content_mask = np.where(
            content_segment.reshape(content_segment.shape[0] * content_segment.shape[1]) == label
        )
        style_mask = np.where(
            style_segment.reshape(style_segment.shape[0] * style_segment.shape[1]) == label
        )
        c_size = content_mask[0].size
        s_size = style_mask[0].size
        if c_size > 10 and s_size > 10 and c_size / s_size < 100 and s_size / c_size < 100:
            label_indicator[label] = True
        else:
            label_indicator[label] = False
    return label_set, label_indicator
