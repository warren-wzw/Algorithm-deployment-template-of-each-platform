#include "main.h"
#include "acl/acl.h"

#define INFO_LOG(fmt, ...)  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...)  fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR]  " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;
bool g_isDevice = false;
const char *aclConfigPath = "../src/acl.json";
const char *MODEL_PATH="../model/yolov5s.om";
const char *PIC="../data/test.bin";
const int loop_count=11;

static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}
uint32_t load_data(void *&inputBuff)
{
    std::ifstream input_file(PIC, std::ifstream::binary);
    if (input_file.is_open() == false) {
        ERROR_LOG("open file %s failed", PIC);
    }
    input_file.seekg(0, input_file.end);
    uint32_t file_szie = input_file.tellg();
    if (file_szie == 0) {
        ERROR_LOG("binfile is empty, filename is %s", PIC);
        input_file.close();
    }
    cout<<"------------>file size "<<file_szie<<endl;
    input_file.seekg(0, input_file.beg);
     input_file.read(static_cast<char *>(inputBuff), file_szie);
    input_file.close();
    return  file_szie;
}

int main()
{
/***************************************************/
/*****************Init ACL**************************/
/***************************************************/
    cout<<"->ACL INIT "<<endl;
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl init failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
/***************************************************/
/*****************apply resource********************/
/***************************************************/
    // set device only one device
    int32_t deviceId_=0; 
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl set device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    cout<<"->set device "<<deviceId_<<endl;
    // create context (set current)
    cout<<"->create context"<<endl; 
    aclrtContext context_; 
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create context failed, deviceId = %d, errorCode = %d",
            deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    // create stream
    cout<<"->create stream"<<endl;  
    aclrtStream stream_;
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create stream failed, deviceId = %d, errorCode = %d",
            deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    // get run mode
    cout<<"->get run mode"<<endl; 
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    g_isDevice=(runMode==ACL_DEVICE) ;
    
/***************************************************/
/********load model and get infos of model**********/
/***************************************************/
    uint32_t modelId_=0;
    ret = aclmdlLoadFromFile(MODEL_PATH,&modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
            MODEL_PATH, static_cast<int32_t>(ret));
        return FAILED;
    }
    cout<<"->load mode "<<"\""<<MODEL_PATH<<"\""<<" model id is "<<modelId_<<endl; 
    //get model describe
    cout<<"->create model describe"<<endl; 
    aclmdlDesc *modelDesc_;
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }
    cout<<"->get model describe"<<endl; 
    ret = aclmdlGetDesc(modelDesc_, modelId_); 
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    deviceId_=0;
/***************************************************/
/******************prepare input data buffer********/
/***************************************************/    
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    aclmdlDataset *input_;
    void * inputDataBuffer = nullptr;
    size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    cout<<"->get input size "<<modelInputSize<<endl;

    cout<<"->apply input mem "<<endl;
    aclError aclRet = aclrtMalloc(&inputDataBuffer, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu, errorCode is %d",
            modelInputSize, static_cast<int32_t>(aclRet));
        return FAILED;
    }
    uint32_t file_size=load_data(inputDataBuffer);
    if(file_size!=modelInputSize){
        cout<<"->file size not equal model input size!\n "<<endl; 
        return FAILED;
    }
    cout<<"->create input dataset "<<endl;
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    cout<<"->create databuffer"<<endl; 
    aclDataBuffer *inputData = aclCreateDataBuffer(inputDataBuffer, modelInputSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }
    cout<<"->add data to datasetbuffer"<<endl;
    ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    INFO_LOG("create model input success");

/***************************************************/
/************prepare output data buffer*************/
/***************************************************/
    aclmdlDataset *output_;
    cout<<"->create dataset"<<endl;
    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }
    size_t output_num= aclmdlGetNumOutputs(modelDesc_); 
    cout<<"->get num of output "<<output_num<<endl;
    for (size_t i = 0; i < output_num; ++i) {
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        cout<<"-> output size["<<i<<"] :"<<modelOutputSize<<endl;
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                modelOutputSize, static_cast<int32_t>(ret));
            return FAILED;
        }
        //apply output buffer
        cout<<"->apply output buffer"<<endl;
        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            (void)aclrtFree(outputBuffer);
            return FAILED;
        }
        cout<<"->AddDatasetBuffer"<<endl;
        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d",
                static_cast<int32_t>(ret));
            (void)aclrtFree(outputBuffer);
            (void)aclDestroyDataBuffer(outputData);
            return FAILED;
        }

        cout<<"-> get original output test"<<endl;
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
        cout<<"-> getDataBufferSizeV2["<<i<<"] :"<<len<<endl;    
        float *outData = NULL;  
        outData = reinterpret_cast<float*>(data); 
        for(int num=0;num<10;num++){
            cout<<outData[num]<<endl;
        }
    }
    cout<<"->create model output success "<<endl;
    
/***************************************************/
/******************inference************************/
/***************************************************/
    cout<<"->begin inference "<<"model id is "<<modelId_<<endl;
    int64_t sum=0;
    int64_t start_time=0;
    int64_t end_time=0;
    int64_t eclipes_time=0; 
    for(int i = 0; i < loop_count;i++){
         start_time = getCurrentTimeUs();
        //ret = aclmdlExecuteAsync(modelId_, input_, output_,stream_);
        ret = aclmdlExecute(modelId_, input_, output_);
        if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    } 
        end_time = getCurrentTimeUs();
        eclipes_time=end_time-start_time;
        if(i!=0){
            sum =sum + eclipes_time;
        }
        printf("------------------use time %.2f ms\n", eclipes_time/1000.f);
    }
    printf("loop %d averages use time %.2f ms\n",loop_count-1,(sum/1000.f)/(loop_count-1));
    
/***************************************************/
/******************post process*********************/
/***************************************************/
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        cout<<"->get output "<<endl;
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
        cout<<"-> getDataBufferSizeV2["<<i<<"] :"<<len<<endl;    
        float *outData = NULL;  
        outData = reinterpret_cast<float*>(data); 
        for(int num=0;num<10;num++){
            cout<<outData[num]<<endl;
        }
    }
/***************************************************/
/*********************destroy model output*********/
/***************************************************/
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
    INFO_LOG("destroy model output success");

/***************************************************/
/*******************destroy model input*************/
/***************************************************/
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
    INFO_LOG("destroy model input success");

/***************************************************/
/******uninstall model and release resource*********/
/***************************************************/
    cout<<"->unload model id is "<<modelId_<<endl;
    ret = aclmdlUnload(modelId_);
     if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    } 
    INFO_LOG("unload model success, modelId is %u", modelId_);
    // releasemodelDesc_
    if (modelDesc_ != nullptr) {
        aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
    INFO_LOG("release modelDesc_ success, modelId is %u", modelId_);
    //release resorce
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy stream failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        stream_ = nullptr;
    }
    cout<<"->destroy stream done"<<endl;

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy context failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        context_ = nullptr;
    }
    cout<<"->destroy context done "<<endl;
    
    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("reset device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
    }
    cout<<"->reset device id is "<<deviceId_<<endl;

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("  failed, errorCode = %d", static_cast<int32_t>(ret));
    }
    INFO_LOG("end to finalize acl");
}