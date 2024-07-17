import os
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import time
import torch

import amct_onnx as amct

PATH = os.path.realpath('./')
DATA_DIR = os.path.join(PATH, 'data')
PARSER = argparse.ArgumentParser(description='amct_onnx MNIST quantization sample.')
ARGS = PARSER.parse_args()
OUTPUTS = os.path.join(PATH, 'outputs/calibration')
TMP = os.path.join(OUTPUTS, 'tmp')


def onnx_forward(onnx_model, batch_size=1, iterations=100):
    ort_session = ort.InferenceSession(onnx_model, amct.AMCT_SO)

    inference_time =[0]
    input_name = ort_session.get_inputs()[0].name
    #load data
    # inference
    start_time = time.time()
    output = ort_session.run(None, {input_name: input_data})
    end_time = time.time()
    inference_time.append(end_time - start_time)
    # post process
    output = torch.tensor(output[0])  # 将输出转换为 PyTorch 张量
    # 输出结果处理和后续操作...
    pred =np.argmax(output)
    print("------------------------use time ",inference_time[0]*1000,"ms")

def main():
    model_file = './model/model.onnx'
    print('[INFO] Do original model test:')
    onnx_forward(model_file,1,1)
    config_json_file = os.path.join(TMP, 'config.json')
    skip_layers = []
    amct.create_quant_config(
            config_file=config_json_file, model_file=model_file, skip_layers=skip_layers, batch_num=1,
            activation_offset=True, config_defination=None)
    # Phase1: do conv+bn fusion, weights calibration and generate
    #         calibration model
    scale_offset_record_file = os.path.join(TMP, 'record.txt')
    modified_model = os.path.join(TMP, 'modified_model.onnx')
    amct.quantize_model(
        config_file=config_json_file, model_file=model_file, modified_onnx_file=modified_model,
        record_file=scale_offset_record_file)
    onnx_forward(modified_model, 32, 1)
    # Phase3: save final model, one for onnx do fake quant test, one
    #         deploy model for ATC
    result_path = os.path.join(OUTPUTS, 'MNIST')
    amct.save_model(modified_model, scale_offset_record_file, result_path)
    # Phase4: run fake_quant model test
    print('[INFO] Do quantized model test:')
    onnx_forward('%s_%s' % (result_path, 'fake_quant_model.onnx'), 1, 1)
if __name__ == '__main__':
    main()
