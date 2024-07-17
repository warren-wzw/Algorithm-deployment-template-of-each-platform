#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1
#include <stdio.h>
#include <sail/cvwrapper.h>
#include <iostream>
#include <string>
#include <numeric>
#include <sys/time.h>
#include <cstdlib>
//sophon api
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "engine.h"
//opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;
using namespace sail;

const std::string& bmodel_path="./model/test_output_fp32_1b.bmodel";
//const std::string& bmodel_path="./model/test_output_fp16_1b.bmodel";
//const std::string& bmodel_path="./model/test_output_int8_1b.bmodel";
const char * Pic_file="./data/bus.jpg";
const int loop_count=10;

static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}
void dump_shape(std::vector<int> shape)
{
    cout<<"[  ";
    for (const int& value : shape) {
        std::cout << value << " ";
    }
     cout<<"]"<<endl;
}

bool inference(int device_id)
{
    // init Engine
    sail::Engine engine(device_id);    
    // load bmodel without builtin input and output tensors
    engine.load(bmodel_path);
    // get model info
    auto graph_name = engine.get_graph_names().front();
    auto input_name = engine.get_input_names(graph_name).front();
    std::vector<int> input_shape = {1, 3, 640, 640};
    std::map<std::string, std::vector<int>> input_shapes;
    input_shapes[input_name] = input_shape;
    auto input_dtype = engine.get_input_dtype (graph_name, input_name);
    //output 
    auto output_name = engine.get_output_names(graph_name);
    auto output_dtype = engine.get_output_dtype(graph_name, output_name[0]);
    auto output_shape = engine.get_output_shape(graph_name,output_name[0]);
    cout<<"----------------graph_name is "<<graph_name<<endl;
    cout<<"----------------input_dtype is "<<input_dtype<<endl;
    cout<<"----------------output_dtype is "<<output_dtype<<endl;
    cout<<"----------------input_name is "<<input_name<<endl;
    cout<<"----------------output_name is "<<endl;
    for(const string& value :output_name){
        std::cout << value << " "<<endl;
    }
    cout<<"input shape is "<<endl;
    dump_shape(input_shape);
    cout<<"output shape is "<<endl;
    dump_shape(output_shape);

    // get handle to create input and output tensors
    sail::Handle handle = engine.get_handle();
    // allocate input and output tensors with both system and device memory
    cout<<"----------------input_dtype is "<<input_dtype<<endl;
    sail::Tensor in(handle, input_shape, input_dtype, true, true);
    sail::Tensor out(handle, output_shape, output_dtype, true, true);

    std::map<std::string, sail::Tensor*> input_tensors = {{input_name, &in}};
    std::map<std::string, sail::Tensor*> output_tensors= {{output_name[0], &out}};
    float* input = nullptr;
    float* output = nullptr;
    int in_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                    1, std::multiplies<int>());
    int out_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                    1, std::multiplies<int>());
    cout<<"in_size is "<<in_size<<" out_size is "<<out_size<<endl;
    
    if (input_dtype == BM_FLOAT32) {
        input = reinterpret_cast<float*>(in.sys_data());
    } 
    else {
        input = new float[in_size];
    }
    if (output_dtype == BM_FLOAT32) {
        output = reinterpret_cast<float*>(out.sys_data());
    } 
    else {
        output = new float[out_size];
    }
    //load data 
    cv::Mat *src_img=new cv::Mat;
    *src_img=cv::imread(Pic_file);
    cout<<"Pic_file"<<Pic_file<<endl;
    if ((*src_img).empty()) {
        std::cerr << "can not read image!" << std::endl;
        return -1;
    }
    //post process
    sail::Handle handle_1(device_id);
    sail::Bmcv bmcv(handle_1);
    sail::BMImage bmimg;
    bmcv.mat_to_bm_image(*src_img, bmimg);
    cout<<"width "<<bmimg.width()<<"height "<<bmimg.height()<<"bmimg dtype is "<<bmimg.dtype()<<endl;
    //set input
    printf("->set input\n");
    sail::Tensor bmarray=bmcv.bm_image_to_tensor(bmimg);
    in.reset_sys_data(bmarray.sys_data(),input_shape);
    cout<<"in dtype is "<<in.dtype()<<endl;

    // set io_mode SYSO:Both input and output tensors are in system memory.
    engine.set_io_mode(graph_name, sail:: SYSIO);
    //inference
    int64_t sum=0;
    int64_t start_time=0;
    int64_t end_time=0;
    int64_t eclipes_time=0; 
    for(int i = 0; i < loop_count;i++){
        printf("->inferences\n");
        int64_t start_time = getCurrentTimeUs();
        engine.process(graph_name, input_tensors, input_shapes, output_tensors);
        int64_t end_time = getCurrentTimeUs();
        eclipes_time=end_time-start_time;
            sum =sum + eclipes_time;
            printf("------------------use time %.2f ms\n", eclipes_time/1000.f);
    }
    printf("loop %d averages use time %.2f ms\n",loop_count,(sum/1000.f)/loop_count);
    //get output data
    auto real_output_shape_1 = engine.get_output_shape(graph_name,output_name[0]);
    cout<<"output shape is "<<endl;
    dump_shape(real_output_shape_1);
    float* output_data = reinterpret_cast<float*>(out.sys_data());
    //post process
    return true;
}
int main()
{
    int device_id = 0;
    int tpu_num=get_available_tpu_num();
    printf("--------the tpu number is %d\n", tpu_num);
    inference(device_id);
    return 0;    
}