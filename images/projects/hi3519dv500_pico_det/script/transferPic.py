#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (C) Shenshu Technologies Co., Ltd. 2022-2022. All rights reserved.
import numpy as np
import os
from PIL import Image

def process(input_path):
    try:
        input_image = Image.open(input_path)
        input_image = input_image.resize((256, 256), resample=Image.BILINEAR)
        # hwc
        img = np.array(input_image)
        height = img.shape[0]
        width = img.shape[1]
        h_off = int((height-224)/2)
        w_off = int((width-224)/2)
        crop_img = img[h_off:height-h_off, w_off:width-w_off, :]
        # rgb to bgr
        img = crop_img[:, :, ::-1]
        shape = img.shape
        img = img.astype("int8")
        img = img.reshape([1] + list(shape))
        result = img.transpose([0, 3, 1, 2])
        output_name = input_path.rsplit('.', 1)[0] + ".bin"
        result.tofile(output_name)
    except Exception as except_err:
        print(except_err)
        return 1
    else:
        return 0
if __name__ == "__main__":
    count_ok = 0
    count_ng = 0
    images = os.listdir(r'./')
    dir = os.path.realpath("./")
    for image_name in images:
        if not (image_name.lower().endswith((".bmp", ".dib", ".jpeg", ".jpg", ".jpe",
        ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"))):
            continue
        print("start to process image {}....".format(image_name))
        image_path = os.path.join(dir, image_name)
        ret = process(image_path)
        if ret == 0:
            print("process image {} successfully".format(image_name))
            count_ok = count_ok + 1
        elif ret == 1:
            print("failed to process image {}".format(image_name))
            count_ng = count_ng + 1
    print("{} images in total, {} images process successfully, {} images process failed"
          .format(count_ok + count_ng, count_ok, count_ng))
