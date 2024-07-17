'''
Author: warren 2016426377@qq.com
Date: 2023-08-09 15:14:53
LastEditors: warren 2016426377@qq.com
LastEditTime: 2023-10-26 21:48:55
FilePath: \7.26_v1.0c:\00-document\研究生_项目\00-存档文件\00-各平台算法部署\RK3588\template\模型转换\rknn_toolkit\00-template\export_rknn.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from rknn.api import RKNN

ONNX_MODEL = 'yolov5s.onnx'
RKNN_MODEL = 'yolov5s_quat.rknn'
DATASET = './dataset.txt'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    ret=rknn.config(mean_values=[[0, 0, 0]], std_values=[[0, 0, 0]],target_platform='rk3588')  #wzw
    if ret != 0:
        print('config model failed!')
        exit(ret)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['output', '345', '346'])  
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    #ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    #release rknn
    rknn.release()