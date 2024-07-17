import os
import sys
os.chdir(sys.path[0])
import onnxruntime
import torch
import torchvision
import numpy as np
import time
import cv2
sys.path.append('..')
from ultralytics.utils.plotting import Annotator, colors
import struct

ONNX_MODEL="./yolov5n.onnx"
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def xywh2xyxy(x):
    coord=[]
    for x_ in x:
        xl=x_[0]-x_[2]/2
        yl=x_[1]-x_[3]/2
        xr=x_[0]+x_[2]/2
        yr=x_[1]+x_[3]/2
        coord.append([xl,yl,xr,yr])
    coord=torch.tensor(coord).to(x.device)
    return coord

def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    count_true = torch.sum(xc.type(torch.int))

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def draw_bbox(image, result, color=(0, 0, 255), thickness=2):
    # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    image = image.copy()
    for point in result:
        x1,y1,x2,y2=point
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def postprocess(output,obj_conf=0.25,cls_conf=0.45):
    device = output.device
    """first filt"""
    FirstFilterData=[]
    FirstFilter = output[..., 4] > obj_conf  # candidates
    for FirstFilter_,output_ in zip(FirstFilter[0].tolist(),output[0].tolist()):
        if FirstFilter_==True:
            FirstFilterData.append(output_)
    """second filt"""
    FirstFilterData=torch.tensor(FirstFilterData).to(device)
    ObjConf=FirstFilterData[:,4]
    ClsConf=FirstFilterData[:,5:]
    ClsConf = np.array(ClsConf.cpu())
    max_values_per_row = np.max(ClsConf, axis=1)
    max_values_per_row=torch.tensor(max_values_per_row).to(device)
    second=max_values_per_row*ObjConf
    indices = (second > cls_conf).nonzero()

    print()

def main():
    """preprocess"""
    image=cv2.imread('../data/images/bus.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_array=np.array(image) / 255.0
    input_array = np.expand_dims(input_array, axis=0)
    input_array = np.transpose(input_array, (0, 3, 1, 2)).astype(np.float32)
    with open("./pic.bin",'rb') as pic_file:
        data=pic_file.read()
        num_floats = len(data) // 4
        # 解析单精度浮点数
        float_values = struct.unpack('f' * num_floats, data)
        # 打印浮点数值
    
    """"""
    onnx_model = onnxruntime.InferenceSession(ONNX_MODEL)
    input_name = onnx_model.get_inputs()[0].name
    """onnx"""
    out = onnx_model.run(None, {input_name:input_array})
    out_tensor = torch.tensor(out).to(DEVICE)
    """dv500"""
    # out_dv500 = np.fromfile('output.bin', dtype=np.float32)
    # out_tensor = torch.tensor(out_dv500).to(DEVICE)
    # out_tensor=out_tensor.reshape(1,25200,88)
    # out_tensor=out_tensor[:,:,:85]
    """"""
    #pred=postprocess(out_tensor)
    pred = non_max_suppression(out_tensor,0.25,0.45,classes=None,agnostic=False,max_det=1000)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0_=cv2.imread('../data/images/bus.jpg')
        im0=im0_.reshape(1,3,640,640)
        names=torch.load('name.pt')
        annotator = Annotator(im0, line_width=3, example=str(names))
        coord=[]
        image=im0.reshape(640,640,3)
        if len(det):
            """Write results"""
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                c = int(cls)  # integer class
                label = f"{names[c]} {conf:.2f}"
                coord.append([int(xyxy[0].item()), int(xyxy[1].item()),int(xyxy[2].item()), int(xyxy[3].item())])
        image=draw_bbox(image,coord)
        # Stream results
        save_success =cv2.imwrite('result.jpg', image)
        print(f"save image end {save_success}")
               
if __name__=="__main__":
    main()
