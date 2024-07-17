/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "rknn_api.h"

using namespace std;
using namespace cv;

int ret=0;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}
static void dump_tensor_attr(rknn_tensor_attr* attr)  //dump tensor message
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
static unsigned char *load_model(const char *filename, int *model_size) //load model
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
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
        dump_tensor_attr(&(attrs[i]));
    }
}
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{      
    rknn_context ctx;
    int model_len = 0;
    unsigned char *model;
    rknn_output outputs[1];
    rknn_input inputs[1];
    const char *model_path = argv[1];
    if (argc != 2)
    {
        printf("Usage: %s <rknn model>  \n", argv[0]);
        return -1;
    }
    // Load RKNN Model
    cout<<"->load rknn model"<<endl;
    model = load_model(model_path, &model_len);
    cout<<"->done "<<endl;
    cout<<"->init rknn "<<endl;
    ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->done "<<endl;
    // Get Model Input and Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"model input num: "<<io_num.n_input<<", output num: "<<io_num.n_output<<endl;
    //get input tensor message
    cout<<"->input tensors:"<<endl;
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    get_tensor_message(ctx,input_attrs,io_num.n_input,1);
    //get output tensor message
    cout<<"->output tensors:"<<endl;
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    get_tensor_message(ctx,output_attrs,io_num.n_output,0);
    //load data  
    cout<<"->load data:"<<endl;  
    // Set Input Data
    cout<<"->set input data:"<<endl;
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = image.cols*image.rows*image.channels();
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = image.data;
    ret = rknn_inputs_set(ctx, 1, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->done"<<endl;           
    // Run 
    cout<<"->run rknn"<<endl;   
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }   
    cout<<"->done"<<endl; 
    // Get Output
    cout<<"->get output"<<endl;
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->done"<<endl; 
    // Post Process
    cout<<"->post process"<<endl;  
    // Release rknn_outputs
    cout<<"->release output"<<endl;
    ret=rknn_outputs_release(ctx, io_num.n_output, outputs);
    if(ret < 0) {
        printf("rknn_outputs_release fail! ret=%d\n", ret);
        return -1;
    }
    cout<<"->done"<<endl; 
    // Release RKNN
    cout<<"->release rknn"<<endl; 
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    cout<<"->done"<<endl;
    return 0;
}

