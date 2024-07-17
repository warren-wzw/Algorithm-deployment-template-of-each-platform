from rknn.api import RKNN

ONNX_MODEL = 'model.onnx'
RKNN_MODEL = 'model_int8.rknn'

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    rknn.list_devices()

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]],target_platform=["rv1126"])
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export xxxx.rknn failed!')
        exit(ret)
    print('done')
    rknn.release()
