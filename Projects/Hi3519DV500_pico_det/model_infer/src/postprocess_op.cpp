#include "postprocess_op.h"

namespace PaddleOCR {

void PicodetPostProcessor::init(std::string label_path,const double score_threshold,const double nms_threshold,const std::vector<int> &fpn_stride) {
  this->label_list_ = Utility::ReadDict(label_path);
  this->score_threshold_ = score_threshold;
  this->nms_threshold_ = nms_threshold;
  this->num_class_ = label_list_.size();
  this->fpn_stride_ = fpn_stride;
}

void PicodetPostProcessor::Run(std::vector<StructurePredictResult> &results,std::vector<std::vector<float>> outs,std::vector<int> ori_shape,std::vector<int> resize_shape, int reg_max) {
  int in_h = resize_shape[0];
  int in_w = resize_shape[1];
  float scale_factor_h = resize_shape[0] / float(ori_shape[0]);
  float scale_factor_w = resize_shape[1] / float(ori_shape[1]);
  
  std::vector<std::vector<StructurePredictResult>> bbox_results;
  bbox_results.resize(this->num_class_);
  //int num=0;
  for (size_t i = 0; i < this->fpn_stride_.size(); ++i) {
    int feature_h = std::ceil((float)in_h / this->fpn_stride_[i]);
    int feature_w = std::ceil((float)in_w / this->fpn_stride_[i]);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
      float score = 0;
      int cur_label = 0;
      for (int label = 0; label < this->num_class_; label++) {
        if (outs[i][idx * this->num_class_ + label] > score) {
          score = outs[i][idx * this->num_class_ + label];
          cur_label = label;
        }
      }
      // bbox
      if (score > this->score_threshold_) {
          int row = idx / feature_w;
          int col = idx % feature_w;
          std::vector<float> bbox_pred(outs[i + this->fpn_stride_.size()].begin() + idx * 4 * reg_max,outs[i + this->fpn_stride_.size()].begin() +(idx + 1) * 4 * reg_max);
          //num++;
          bbox_results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, col, row,this->fpn_stride_[i], resize_shape, reg_max));
      }
    }
  }
  for (size_t i = 0; i < bbox_results.size(); i++) {
    if (bbox_results[i].size() <= 0) {
      continue;
    }
    this->nms(bbox_results[i], this->nms_threshold_);
    for (auto box : bbox_results[i]) {
      box.box[0] = box.box[0] / scale_factor_w;
      box.box[2] = box.box[2] / scale_factor_w;
      box.box[1] = box.box[1] / scale_factor_h;
      box.box[3] = box.box[3] / scale_factor_h;
      results.push_back(box);
    }
  }
}

StructurePredictResult PicodetPostProcessor::disPred2Bbox(std::vector<float> bbox_pred, int label,float score, int x, int y, int stride,std::vector<int> im_shape, int reg_max) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    std::vector<float> bbox_pred_i(bbox_pred.begin() + i * reg_max,bbox_pred.begin() + (i + 1) * reg_max);
    std::vector<float> dis_after_sm =Utility::activation_function_softmax(bbox_pred_i);
    for (int j = 0; j < reg_max; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
  }

  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)im_shape[1]);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)im_shape[0]);
  //std::cout<<" xmin "<<xmin<<" ymin "<<ymin<<" xmax "<<xmax<<" ymax "<<ymax<<std::endl;
  StructurePredictResult result_item;
  result_item.box=std::vector<float>{xmin, ymin,xmax,ymax};
  (result_item).confidence = score;
  (result_item).type = this->label_list_[label];
  //std::cout<<"---------(result_item).type is "<<(result_item).type<<std::endl;
  //std::cout<<"result_item "<<(result_item).box[0]<<" "<<(result_item).box[1]<<" "<<(result_item).box[2]<<" "<<(result_item).box[3]<<" "<<std::endl;
  return (result_item);
}

void PicodetPostProcessor::nms(std::vector<StructurePredictResult> &input_boxes,
                               float nms_threshold) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](StructurePredictResult a, StructurePredictResult b) {
              return a.confidence > b.confidence;
            });
  std::vector<int> picked(input_boxes.size(), 1);

  for (size_t i = 0; i < input_boxes.size(); ++i) {
    if (picked[i] == 0) {
      continue;
    }
    for (size_t j = i + 1; j < input_boxes.size(); ++j) {
      if (picked[j] == 0) {
        continue;
      }
      float iou = Utility::iou(input_boxes[i].box, input_boxes[j].box);
      if (iou > nms_threshold) {
        picked[j] = 0;
      }
    }
  }
  std::vector<StructurePredictResult> input_boxes_nms;
  for (size_t i = 0; i < input_boxes.size(); ++i) {
    if (picked[i] == 1) {
      input_boxes_nms.push_back(input_boxes[i]);
    }
  }
  input_boxes = input_boxes_nms;
}

} // namespace PaddleOCR
