/*******************************************************************************
 * File Name : template_main.cpp
 * Version : 0.0
 * Author : warren_wzw
 * Created : 2023.09.15
 * Last Modified :
 * Description :  
 * Function List :
 * Modification History :
******************************************************************************/
#include "main.h"
#include <stdio.h>

const char *aclConfigPath = "../src/acl.json";  
const char *OUTPUT_BIN = "../data/output.bin";
const int CLSNUM = 83 ;//stride is 88 for mem align so 88-5=83
const char *IMAGENAME="../data/bus.jpg";


static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

template <typename T>
void load_data(T *inputBuff,const char *file)
{
    std::ifstream input_file(file, std::ifstream::binary);
    if (input_file.is_open() == false) {
        ERROR_LOG("open file %s failed", file);
    }
    input_file.seekg(0, input_file.end);
    uint32_t file_size = input_file.tellg();
    if (file_size == 0) {
        ERROR_LOG("binfile is empty, filename is %s", file);
        input_file.close();
    }
    cout<<"->file size is "<<file_size<<endl;
    input_file.seekg(0, input_file.beg);
    input_file.read(reinterpret_cast<char*>(inputBuff), file_size);
    input_file.close();
}

void LoadDataC(float * Buffer,const char * filename){
    FILE *fp=NULL;
    long int size;
    /*open file*/
    fp=fopen(filename,"r");
    if(fp==NULL){
        printf("open file fail !!\n");
    }
    /*get file size*/
    fseek(fp,0,SEEK_END);
    size=ftell(fp);
    fseek(fp,0,SEEK_SET);
    printf("file size is %ld data num is %ld\n",size,size/sizeof(float));
    /*read file*/
    int read_status=fread(Buffer,sizeof(float),size/sizeof(float),fp);
    if (read_status!=1){
        printf("read file failed !\n");
    }
    /*close*/
    fclose(fp);
}

void WritedDataC(float * Buffer,const char * filename,int size)
{
    FILE * fp=NULL;
    fp=fopen(filename,"w");
    if (fp ==NULL){
        printf("fail to open file !\n");
    }
    fwrite(Buffer,sizeof(float),size,fp);
}

float iou(Bbox box1, Bbox box2) {
    /*  
    iou=交并比
    */
    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.w, box2.x + box2.w);
    int y2 = min(box1.y + box1.h, box2.y + box2.h);
    int w = max(0, x2 - x1);
    int h = max(0, y2 - y1);
    float over_area = w * h;
    return over_area / (box1.w * box1.h + box2.w * box2.h - over_area);
}

bool judge_in_lst(int index, vector<int> index_lst) {
    //若index在列表index_lst中则返回true，否则返回false
    if (index_lst.size() > 0) {
        for (int i = 0; i < int(index_lst.size()); i++) {
            if (index == index_lst.at(i)) {
                return true;
            }
        }
    }
    return false;
}

int get_max_index(vector<Detection> pre_detection) {
    //返回最大置信度值对应的索引值
    int index;
    float conf;
    if (pre_detection.size() > 0) {
        index = 0;
        conf = pre_detection.at(0).conf;
        for (int i = 0; i < int(pre_detection.size()); i++) {
            if (conf < pre_detection.at(i).conf) {
                index = i;
                conf = pre_detection.at(i).conf;
            }
        }
        return index;
    }
    else {
        return -1;
    }
}

vector<int> nms(vector<Detection> pre_detection, float iou_thr)
{
    /*
    返回需保存box的pre_detection对应位置索引值
    */
    int index;
    vector<Detection> pre_detection_new;
    //Detection det_best;
    Bbox box_best, box;
    float iou_value;
    vector<int> keep_index;
    vector<int> del_index;
    bool keep_bool;
    bool del_bool;

    if (pre_detection.size() > 0) {
        pre_detection_new.clear();
        // 循环将预测结果建立索引
        for (int i = 0; i < int(pre_detection.size()); i++) {
            pre_detection.at(i).index = i;
            pre_detection_new.push_back(pre_detection.at(i));
        }
        //循环遍历获得保留box位置索引-相对输入pre_detection位置
        while (pre_detection_new.size() > 0) {
            index = get_max_index(pre_detection_new);
            if (index >= 0) {
                keep_index.push_back(pre_detection_new.at(index).index); //保留索引位置

                // 更新最佳保留box
                box_best.x = pre_detection_new.at(index).bbox[0];
                box_best.y = pre_detection_new.at(index).bbox[1];
                box_best.w = pre_detection_new.at(index).bbox[2];
                box_best.h = pre_detection_new.at(index).bbox[3];

                for (int j = 0; j < int(pre_detection.size()); j++) {
                    keep_bool = judge_in_lst(pre_detection.at(j).index, keep_index);
                    del_bool = judge_in_lst(pre_detection.at(j).index, del_index);
                    if ((!keep_bool) && (!del_bool)) { //不在keep_index与del_index才计算iou
                        box.x = pre_detection.at(j).bbox[0];
                        box.y = pre_detection.at(j).bbox[1];
                        box.w = pre_detection.at(j).bbox[2];
                        box.h = pre_detection.at(j).bbox[3];
                        iou_value = iou(box_best, box);
                        if (iou_value > iou_thr) {
                            del_index.push_back(j); //记录大于阈值将删除对应的位置
                        }
                    }

                }
                //更新pre_detection_new
                pre_detection_new.clear();
                for (int j = 0; j < int(pre_detection.size()); j++) {
                    keep_bool = judge_in_lst(pre_detection.at(j).index, keep_index);
                    del_bool = judge_in_lst(pre_detection.at(j).index, del_index);
                    if ((!keep_bool) && (!del_bool)) {
                        pre_detection_new.push_back(pre_detection.at(j));
                    }
                }
            }
        }
    }

    del_index.clear();
    del_index.shrink_to_fit();
    pre_detection_new.clear();
    pre_detection_new.shrink_to_fit();

    return  keep_index;

}

void PreProcess(float * input_buf)
{
    cv::Mat image = cv::imread("../data/bus.jpg");
    if (image.empty()) {
        std::cerr << "Failed to read image." << std::endl;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // 将像素值归一化到 [0, 1] 范围
    cv::Mat input_array;
    image.convertTo(input_array, CV_32FC3, 1.0 / 255.0);

    int rh = input_array.rows;
    int rw = input_array.cols;
    int rc = input_array.channels();
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(input_array, cv::Mat(rh, rw, CV_32FC1, input_buf + i * rh * rw), i);
    }
    //input_buf = input_array.ptr<float>;
}

vector<Detection> PostProcess(float* prob,float conf_thr=0.3,float nms_thr=0.5)
{
    vector<Detection> pre_results;
    vector<int> nms_keep_index;
    vector<Detection> results;
    bool keep_bool;
    Detection pre_res;
    float conf;
    int tmp_idx;
    float tmp_cls_score;
    for (int i = 0; i < 25200; i++) {
        tmp_idx = i * (CLSNUM + 5);
        pre_res.bbox[0] = prob[tmp_idx + 0];  //cx
        pre_res.bbox[1] = prob[tmp_idx + 1];  //cy
        pre_res.bbox[2] = prob[tmp_idx + 2];  //w
        pre_res.bbox[3] = prob[tmp_idx + 3];  //h
        conf = prob[tmp_idx + 4];  // 是为目标的置信度
        tmp_cls_score = prob[tmp_idx + 5] * conf; //conf_thr*nms_thr
        pre_res.class_id = 0;
        pre_res.conf = 0;
        // 这个过程相当于从除了前面5列，在后面的cla_num个数据中找出score最大的值作为pre_res.conf，对应的列作为类id
        for (int j = 1; j < CLSNUM; j++) {     
            tmp_idx = i * (CLSNUM + 5) + 5 + j; //获得对应类别索引
            if (tmp_cls_score < prob[tmp_idx] * conf){
                tmp_cls_score = prob[tmp_idx] * conf;
                pre_res.class_id = j;
                pre_res.conf = tmp_cls_score;
            }
        }
        if (conf >= conf_thr) {
            pre_results.push_back(pre_res);
        }
    }
    //使用nms，返回对应结果的索引
    nms_keep_index=nms(pre_results,nms_thr);
    // 茛据nms找到的索引，将结果取出来作为最终结果
    for (int i = 0; i < int(pre_results.size()); i++) {
        keep_bool = judge_in_lst(i, nms_keep_index);
        if (keep_bool) {
            results.push_back(pre_results.at(i));
        }
    }

    pre_results.clear();
    pre_results.shrink_to_fit();
    nms_keep_index.clear();
    nms_keep_index.shrink_to_fit();

    return results; 
}

void drawTextBox(cv::Mat &boxImg,const std::vector<cv::Point> &box, int thickness) {
    auto color = cv::Scalar(0, 0, 255);// R(255) G(0) B(0)
    cv::line(boxImg, box[0], box[1], color, thickness);
    cv::line(boxImg, box[1], box[2], color, thickness);
    cv::line(boxImg, box[2], box[3], color, thickness);
    cv::line(boxImg, box[3], box[0], color, thickness);
}

int main(int argc, char **argv)
{
    svp_acl_error ret;
    const char *MODEL_PATH = argv[1];
    const char *INPUTDATA = argv[2];
    if (argc != 3){
        printf("Usage: ./xxx  model_path  input_data");
    }
/***************************************************/
/*****************Init ACL**************************/
/***************************************************/
    cout<<"->ACL INIT "<<endl;
    // ACL init
    ret = svp_acl_init(aclConfigPath);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");
/***************************************************/
/*****************apply resource********************/
/***************************************************/
    int32_t deviceId_=0;
    ret = svp_acl_rt_set_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);
    // create context (set current)
    svp_acl_rt_context context_=nullptr;
    ret = svp_acl_rt_create_context(&context_, deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");
    // create stream 
    svp_acl_rt_stream stream_=nullptr;
    ret = svp_acl_rt_create_stream(&stream_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");
    // get run mode
    svp_acl_rt_run_mode runMode;
    ret = svp_acl_rt_get_run_mode(&runMode);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    if (runMode != SVP_ACL_DEVICE) {
        ERROR_LOG("acl run mode failed");
        return FAILED;
    }
    INFO_LOG("get run mode success");

/***************************************************/
/********load model and get infos of model**********/
/***************************************************/
    uint32_t modelId_=0;
    std::ifstream input_file(MODEL_PATH, std::ifstream::binary);
    if (input_file.is_open() == false) {
        ERROR_LOG("open file %s failed", MODEL_PATH);
    }
    input_file.seekg(0, input_file.end);
    uint32_t file_size = input_file.tellg();
    void * model_buffer = nullptr;
    svp_acl_rt_malloc(&model_buffer,file_size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);   
    uint8_t * model_buffer_ = static_cast<uint8_t *>(model_buffer);
    load_data(model_buffer_,MODEL_PATH);
    ret = svp_acl_mdl_load_from_mem(static_cast<uint8_t* >(model_buffer_), file_size, &modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        svp_acl_rt_free(model_buffer_);
        ERROR_LOG("load model from file failed, model file is %s", MODEL_PATH);
        return FAILED;
    }
    INFO_LOG("load model %s success model id is %d ", MODEL_PATH,modelId_);
    //get model describe
    svp_acl_mdl_desc *modelDesc_=nullptr;
    modelDesc_ =  svp_acl_mdl_create_desc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }
    ret =  svp_acl_mdl_get_desc(modelDesc_, modelId_); 
    if (ret != SUCCESS) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    //get input info
    svp_acl_mdl_io_dims dims;
    svp_acl_format format;
    svp_acl_data_type data_type;
    size_t stride=0;
    size_t input_num= svp_acl_mdl_get_num_inputs(modelDesc_);
    for(int num=0;num<(int)input_num;num++){
        stride= svp_acl_mdl_get_input_default_stride(modelDesc_, num);
        if (stride == 0) {
            ERROR_LOG("svp_acl_mdl_get_input_default_stride error!");
            return FAILED;
        }
        ret = svp_acl_mdl_get_input_dims(modelDesc_, num,&dims);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("svp_acl_mdl_get_input_dims error!");
            return FAILED;
        }
        format=svp_acl_mdl_get_input_format(modelDesc_,num);
        data_type=svp_acl_mdl_get_input_data_type(modelDesc_,num);
        size_t input_stride= svp_acl_mdl_get_input_default_stride(modelDesc_,num);
        const char *input_name=svp_acl_mdl_get_input_name_by_index(modelDesc_,num);
        std::cout<<"->dims [ ";
        for(int num=0;num<(int)dims.dim_count;num++){
            cout<<dims.dims[num]<<" ";}
        std::cout<<"] ";
        std::cout<<"->input format is "<<format<<" data_type is "<<data_type<<" input_stride is "\
                                                <<input_stride<<" input_name is "<<input_name<<std::endl;
    }
/***************************************************/
/******************prepare input data buffer********/
/***************************************************/ 
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    size_t modelInputSize =  svp_acl_mdl_get_input_size_by_index(modelDesc_, 0);
    cout<<"->get input size "<<modelInputSize<<endl;
    svp_acl_mdl_dataset *input_=nullptr;
    void * inputDataBuffer = nullptr;
    svp_acl_error aclRet = svp_acl_rt_malloc(&inputDataBuffer, modelInputSize*sizeof(float), SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (aclRet != SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu, errorCode is %d",
            modelInputSize, static_cast<int32_t>(aclRet));
        return FAILED;
    }
/******************preprocess*******************/
    float * input_image_float = static_cast<float *>(inputDataBuffer);
    //LoadDataC(input_image_float,INPUTDATA); 
    PreProcess(input_image_float);
    WritedDataC(input_image_float,"../data/pic.bin",modelInputSize/sizeof(float));
    for (int i=0;i<100;i++){
        cout<<input_image_float[i]<<endl;
    }

/******************load data to dataset********/
    input_ =  svp_acl_mdl_create_dataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    stride = svp_acl_mdl_get_input_default_stride(modelDesc_, 0);
    if (stride == 0) {
        ERROR_LOG("Error, output default stride is %lu.", stride);
        return FAILED;
    }
    svp_acl_data_buffer *inputData = svp_acl_create_data_buffer(input_image_float, modelInputSize*4 ,stride);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }
    ret = svp_acl_mdl_add_dataset_buffer(input_, inputData);
    if (ret != SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)svp_acl_destroy_data_buffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    INFO_LOG("create model input success");
/***************************************************/
/************prepare CreateTaskBufAndWorkBuf********/
/***************************************************/
    //  taskbuf and workbuf
    if (input_num<= 2) {
        ERROR_LOG("input dataset Num is error.");
        return FAILED;
    }
    size_t datasetSize = svp_acl_mdl_get_dataset_num_buffers(input_);
    if (datasetSize == 0) {
        ERROR_LOG("input dataset Num is 0.");
        return FAILED;
    }
    void * inputDataBuffer_[input_num-1] = {};
    svp_acl_data_buffer *inputData_[input_num-1]={};
    for (size_t loop = datasetSize; loop < svp_acl_mdl_get_num_inputs(modelDesc_); loop++){
        stride = svp_acl_mdl_get_input_default_stride(modelDesc_, loop);
        modelInputSize =  svp_acl_mdl_get_input_size_by_index(modelDesc_, loop);
        svp_acl_error aclRet = svp_acl_rt_malloc(&inputDataBuffer_[loop-1], modelInputSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
        if (aclRet != SUCCESS) {
            ERROR_LOG("malloc device buffer failed. size is %zu, errorCode is %d",
                modelInputSize, static_cast<int32_t>(aclRet));
            return FAILED;
        }
        inputData_[loop-1]= svp_acl_create_data_buffer(inputDataBuffer_[loop-1], modelInputSize,stride);
        ret = svp_acl_mdl_add_dataset_buffer(input_, inputData_[loop-1]);
        if (ret != SUCCESS) {
            ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
            (void)svp_acl_destroy_data_buffer(inputData);
            inputData = nullptr;
            return FAILED;
        }
        cout<<"--> loop time is "<<loop<<"stride is "<<stride<<"model input size is "<<modelInputSize<<endl;
    }

/***************************************************/
/************prepare output data buffer*************/
/***************************************************/
    svp_acl_mdl_dataset *output_;
    size_t modelOutputSize=0;
    output_ =  svp_acl_mdl_create_dataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }
    size_t output_num= svp_acl_mdl_get_num_outputs(modelDesc_); 
    cout<<"->get num of output "<<output_num<<endl;
    for (size_t i = 0; i < output_num; ++i) {
        modelOutputSize=0;
        modelOutputSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, i);
        void *outputBuffer = nullptr;
        svp_acl_error ret = svp_acl_rt_malloc(&outputBuffer, modelOutputSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                modelOutputSize, static_cast<int32_t>(ret));
            return FAILED;
        }
        stride = svp_acl_mdl_get_output_default_stride(modelDesc_, i);
        if (stride == 0) {
            ERROR_LOG("Error, output default stride is %lu.", stride);
            return FAILED;
        }
        cout<<"-> output ["<<i<<"] size is :"<<modelOutputSize<<"stride is "<<stride<<endl;
        //apply output buffer
        svp_acl_data_buffer *outputData = svp_acl_create_data_buffer(outputBuffer, modelOutputSize,stride);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            (void)svp_acl_rt_free(outputBuffer);
            return FAILED;
        }
        ret = svp_acl_mdl_add_dataset_buffer(output_, outputData);
        if (ret != SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d",static_cast<int32_t>(ret));
            (void)svp_acl_rt_free(outputBuffer);
            (void)svp_acl_destroy_data_buffer(outputData);
            return FAILED;
        }
    }
    INFO_LOG("create model output success ");
/***************************************************/
/******************get input data only test*********/
/***************************************************/
    svp_acl_data_buffer* input_data_ = svp_acl_mdl_get_dataset_buffer(input_, 0);
    void* input_data1 = svp_acl_get_data_buffer_addr(input_data_);
    uint32_t input_len = svp_acl_get_data_buffer_size(input_data_);
    cout<<"-> get inputDataBufferSize["<<0<<"] :"<<input_len<<endl;
    float *inData = nullptr;  
    inData = reinterpret_cast<float*>(input_data1); 
    cout<<"input data"<<endl;
    for(int num=0;num<0;num++){
        printf("%f \n",inData[num]);
        if(num>1459200){
           printf("%f \n",inData[num]);
        }
    }
/***************************************************/
/******************inference************************/
/***************************************************/
    cout<<"->begin inference "<<"model id is "<<modelId_<<endl;
    int64_t start_time=0;
    int64_t end_time=0;
    int64_t eclipes_time=0; 
    start_time = getCurrentTimeUs();
    ret = svp_acl_mdl_execute(modelId_, input_, output_);
    if (ret != SUCCESS) {
    ERROR_LOG("execute model failed, modelId is %u, errorCode is %d",
        modelId_, static_cast<int32_t>(ret));
    return FAILED;
    } 
    end_time = getCurrentTimeUs();
    eclipes_time=end_time-start_time;
    printf("------------------use time %.2f ms\n", eclipes_time/1000.f); 
/***************************************************/
/*******************destroy model input*************/
/***************************************************/
    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(input_); ++i) {
        svp_acl_data_buffer *dataBuffer = svp_acl_mdl_get_dataset_buffer(input_, i);
        (void)svp_acl_destroy_data_buffer(dataBuffer);
    }
    (void)svp_acl_mdl_destroy_dataset(input_);
    input_ = nullptr;
    INFO_LOG("destroy model input success"); 
/***************************************************/
/*******************get out put data*************/
/***************************************************/
    output_num= svp_acl_mdl_get_num_outputs(modelDesc_);  
    std::vector<const char *> output_names;
    std::vector<int> output_shape;
    std::vector<float> out_data;
    std::vector<std::vector<float>> out_tensor_list(output_num);
    std::vector<std::vector<int>> output_shape_list(output_num);

    float *outData = NULL;  
    for(int num=0;num<(int)output_num;num++){
        const char *output_name;
        size_t output_stride;
        uint32_t len ;
        output_shape.clear();
        svp_acl_mdl_get_output_dims(modelDesc_, num,&dims);
        output_name=svp_acl_mdl_get_output_name_by_index(modelDesc_,num);
        output_names.push_back(output_name);
        format=svp_acl_mdl_get_output_format(modelDesc_,num);
        data_type=svp_acl_mdl_get_output_data_type(modelDesc_,num);
        output_stride= svp_acl_mdl_get_output_default_stride(modelDesc_,num);
        svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, num);
        void* data = svp_acl_get_data_buffer_addr(dataBuffer);
        len = svp_acl_get_data_buffer_size(dataBuffer);
        cout<<"->dims [ ";
        for(int dim_num=0;dim_num<(int)dims.dim_count;dim_num++){
            cout<<dims.dims[dim_num]<<" ";
            output_shape.push_back(dims.dims[dim_num]);
        }
        cout<<"] output format is "<<format<<" data_type is"<<data_type<<" output_stride is "
            <<output_stride<<" output_name is "<<output_names[num]<<"-> getDataBufferSize["<<num<<"] :"<<len<<endl; 
        outData = reinterpret_cast<float*>(data);  
        WritedDataC(outData,OUTPUT_BIN,modelOutputSize/sizeof(float));
        printf("wirite data to file size is %ld ,new version \n",modelOutputSize);
        //show output data
        // for(int show_num=0;show_num<100;show_num++){
        //     printf("%f \n",outData[show_num]);
        // }
    }
/***************************************************/
/******************post process*********************/
/***************************************************/ 
    cv::Mat image;
    image=imread(IMAGENAME,cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "can't read image." << std::endl;
        return -1;
    }
    vector<Detection> results= PostProcess(outData);
    cout<<"results size is "<<results.size()<<endl;
    for (const auto& detection : results) {
        std::vector<cv::Point> box_point;
        std::cout << "Bounding Box: (" << detection.bbox[0] << ", " << detection.bbox[1] << ", "
                  << detection.bbox[2] << ", " << detection.bbox[3] << "), "
                  << "Confidence: " << detection.conf << ", "
                  << "Class ID: " << detection.class_id << ", "
                  << "Index: " << detection.index << std::endl;
        float cx=detection.bbox[0];
        float cy=detection.bbox[1];
        float w=detection.bbox[2];
        float h=detection.bbox[3];
        box_point.push_back(cv::Point(cx+w/2,cy-h/2));  // 右上角
        box_point.push_back(cv::Point(cx-w/2,cy-h/2));  // 左上角
        box_point.push_back(cv::Point(cx-w/2,cy+h/2));  // 左下角  
        box_point.push_back(cv::Point(cx+w/2,cy+h/2));  // 右下角 
        drawTextBox(image,box_point,3);
    }
    INFO_LOG("->save out image!");
    std::string out_name="out.jpg";
    cv::imwrite(out_name,image);
/***************************************************/
/*********************destroy model output*********/
/***************************************************/
    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(output_); ++i) {
        svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, i);
        void* data = svp_acl_get_data_buffer_addr(dataBuffer);
        (void)svp_acl_rt_free(data);
        (void)svp_acl_destroy_data_buffer(dataBuffer);
    }
    (void)svp_acl_mdl_destroy_dataset(output_);
    output_ = nullptr;
    INFO_LOG("destroy model output success"); 

/***************************************************/
/******uninstall model and release resource*********/
/***************************************************/
    ret = svp_acl_mdl_unload(modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }
    (void)svp_acl_rt_free(model_buffer);
    model_buffer=nullptr;
    INFO_LOG("unload model success, modelId is %u", modelId_);
    // releasemodelDesc_
    if (modelDesc_ != nullptr) {
            (void)svp_acl_mdl_destroy_desc(modelDesc_);
            modelDesc_ = nullptr;
    }
    //release resorce
    if (stream_ != nullptr) {
        ret = svp_acl_rt_destroy_stream(stream_);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = svp_acl_rt_destroy_context(context_);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = svp_acl_rt_reset_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = svp_acl_finalize();
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}