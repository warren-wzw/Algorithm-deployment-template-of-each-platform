
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <stb/stb_image_resize.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "rknn_api.h"
using namespace std;
using namespace cv;
int ret =0;


/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}
static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}
void get_tensor_message(rknn_context ctx,rknn_tensor_attr *attrs,uint32_t num,int io)
{
    for (int i = 0; i < num; i++) {
        attrs[i].index = i;
        if(io==1){
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(attrs[i]), sizeof(rknn_tensor_attr));
        }
        else{
            ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(attrs[i]), sizeof(rknn_tensor_attr));
        }
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
        }
        printRKNNTensor(&(attrs[i]));
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    rknn_context ctx;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
        if (argc != 2)
    {
        printf("Usage: ./rknn_MNIST ./model_path");
    }
    // Load RKNN Model
    cout<<"->load model"<<model_path<<endl;
    model = load_model(model_path, &model_len);
    cout<<"->rknn init"<<endl;
    ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0){
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->done"<<endl;
    // Get Model Input Output Info
    cout<<"->get model in/output info"<<endl;
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->model input num: "<<io_num.n_input<<", output num:"<<io_num.n_output<<endl;
    cout<<"->input tensors"<<endl;
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    get_tensor_message(ctx,input_attrs,io_num.n_input,1);
    cout<<"->output tensors"<<endl;
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    get_tensor_message(ctx,output_attrs,io_num.n_output,0);
    cout<<"->done"<<endl;
    // Load data
    cout<<"->load data"<<endl;
    // Set Input Data
    cout<<"->set input data"<<endl;
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = input_attrs[0].size;
    inputs[0].fmt = RKNN_TENSOR_NCHW;
    //inputs[0].buf = input_aray;
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0){
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->done"<<endl;
    // Run
    cout<<"->rknn run"<<endl;
    int64_t start_time = getCurrentTimeUs();
    ret = rknn_run(ctx, nullptr);
    int64_t end_time = getCurrentTimeUs();
    if (ret < 0){
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    printf("------------------use time %.2f ms\n", (end_time-start_time)/1000.f);
    // Get Output
    cout<<"->get output"<<endl;
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->done"<<endl;
    //postprocess
    cout<<"->post process"<<endl;
    // Release rknn_outputs
    cout<<"->release"<<endl;
    rknn_outputs_release(ctx, 1, outputs);
    // Release
    if (ctx >= 0){
        rknn_destroy(ctx);
    }
    if (model){
        free(model);
    }
    cout<<"->done"<<endl;
    return 0;
}
