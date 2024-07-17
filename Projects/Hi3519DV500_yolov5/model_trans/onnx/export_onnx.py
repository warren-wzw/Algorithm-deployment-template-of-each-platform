import os
import sys
os.chdir(sys.path[0])
import onnx
import torch
sys.path.append('..')
from models.common import DetectMultiBackend
from models.experimental import attempt_load
DEVICE='cuda' if torch.cuda.is_available else 'cpu'
def main():
    """create model """
    input = torch.randn(1, 3, 640, 640, requires_grad=False).float().to(torch.device(DEVICE))
    model = attempt_load('./model/yolov5n.pt', device=DEVICE, inplace=True, fuse=True)  # load FP32 model
    #model = DetectMultiBackend('./model/yolov5n.pt', data=input)
    model.to(DEVICE)

    torch.onnx.export(model,
            input,
            'yolov5n_self.onnx', # name of the exported onnx model
            export_params=True,
            opset_version=12,
            do_constant_folding=False, 
            input_names=["images"])
if __name__=="__main__":
    main()