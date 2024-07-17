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

const char *aclConfigPath = "../src/acl.json";  
const char *PIC_BIN="../data/image_data.bin";

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

int main(int argc, char **argv)
{
    svp_acl_error ret;
    const char *MODEL_PATH = argv[1];
    const char *IMAGE = argv[2];
    if (argc != 3){
        printf("Usage: ./xxx  model_path  pic_path");
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
    cv::Mat image;
    image=imread(IMAGE,cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "无法读取图像文件." << std::endl;
        return -1;
    }
    float * input_image_float = static_cast<float *>(inputDataBuffer);
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
    output_ =  svp_acl_mdl_create_dataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }
    size_t output_num= svp_acl_mdl_get_num_outputs(modelDesc_); 
    cout<<"->get num of output "<<output_num<<endl;
    for (size_t i = 0; i < output_num; ++i) {
        size_t modelOutputSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, i);
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
        cout<<"] output format is "<<format<<" data_type is"<<data_type<<" output_stride is "<<output_stride<<" output_name is "<<output_names[num]<<"-> getDataBufferSize["<<num<<"] :"<<len<<endl;  
        float *outData = NULL;  
        outData = reinterpret_cast<float*>(data); 
        //show output data
        for(int show_num=0;show_num<0;show_num++){
            printf("%f \n",outData[show_num]);
        }
    }
/***************************************************/
/******************post process*********************/
/***************************************************/  

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