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

#include "preprocess_op.h"
#include "postprocess_op.h"
#include "utility.h"

using namespace std;
using namespace cv;
using namespace PaddleOCR;

#define INFO_LOG(fmt, ...)  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...)  fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR]  " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

static inline int64_t getCurrentTimeUs();
template <typename T>
void load_data(T *inputBuff,const char *file);
