import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

RKNN_MODEL = 'yolov5s.rknn'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)
    #load model
    print('--> Loading model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')
    # Init runtime environment
    print('--> Init runtime environment')
    #ret = rknn.init_runtime()
    ret = rknn.init_runtime('rk3588',device_id='9249b8a0b4e08c45') #wzw
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[ ])
    print('done')
    # post process
    
    rknn.release()
