import numpy as np
import os
from PIL import Image

def process(input_path):
    try:
        input_image = Image.open(input_path)
        input_image = input_image.resize((640, 640))  # Change the size to (640, 640)
        # hwc
        img = np.array(input_image)
        height = img.shape[0]
        width = img.shape[1]
        h_off = int((height - 640) / 2)  # Update the offset calculation for 640x640
        w_off = int((width - 640) / 2)   # Update the offset calculation for 640x640
        crop_img = img[h_off:h_off + 640, w_off:w_off + 640, :]  # Crop to (448, 448) instead of (224, 224)
        # rgb to bgr
        img = crop_img[:, :, ::-1]
        shape = img.shape
        img = img.astype("float32")
        img[:, :, 0] -= 104
        img[:, :, 1] -= 117
        img[:, :, 2] -= 123
        img = img.reshape([1] + list(shape))
        result = img.transpose([0, 3, 1, 2])
        output_name = input_path.split('.')[0] + ".bin"
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
    for image_name in images:
        if not image_name.endswith("jpg"):
            continue
        print("start to process image {}....".format(image_name))
        ret = process(image_name)
        if ret == 0:
            print("process image {} successfully".format(image_name))
            count_ok = count_ok + 1
        elif ret == 1:
            print("failed to process image {}".format(image_name))
            count_ng = count_ng + 1
    if count_ng == 0:
        print("{} images in total, {} images process successfully" .format(count_ok + count_ng, count_ok))
    else:
        print("{} images in total, {} images process successfully, {} images process failed"
            .format(count_ok + count_ng, count_ok, count_ng))
