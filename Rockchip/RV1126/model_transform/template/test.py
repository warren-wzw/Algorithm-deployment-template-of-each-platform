import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = 'model.onnx'
RKNN_MODEL = 'model_int8.rknn'

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    rknn.list_devices()

    #load rknn model
    ret = rknn.load_rknn(path=RKNN_MODEL)
    if ret != 0:
        print('load rknn failed')
        exit(ret)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rv1126', device_id='cde19cf0f9bf6103',perf_debug=True,eval_mem=True)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    # Set inputs
    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[  ])
    #post process
    
    rknn.release()
