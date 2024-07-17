/*******************************************************************************
 * File Name : template_main.h
 * Version : 0.0
 * Author : warren_wzw
 * Created : 2023.09.15
 * Last Modified :
 * Description :  
 * Function List :
 * Modification History :
******************************************************************************/
#ifndef __MAIN_H_
#define __MAIN_H_

#include <iostream>
#include <map>
#include <sstream>
#include <algorithm>
#include <functional>
#include <sys/stat.h>
#include <fstream>
#include <cstring>
#include <sys/time.h>

#include "acl/svp_acl.h"
#include "acl/svp_acl_mdl.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;


#define INFO_LOG(fmt, ...)  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...)  fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR]  " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

struct  Detection {
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    int class_id;
    int index;
};
struct Bbox {
    int x;
    int y;
    int w;
    int h;
};

static inline int64_t getCurrentTimeUs();
template <typename T>
void load_data(T *inputBuff,const char *file);

#endif // !__MAIN_H