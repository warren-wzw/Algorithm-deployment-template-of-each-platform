set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

include_directories("/home/wzw/soc-sdk/include")
include_directories("/home/wzw/soc-sdk/include/sail")
include_directories("/home/wzw/soc-sdk/include/opencv4/")
# 添加交叉编译需要使用的头文件目录

link_directories("/home/wzw/soc-sdk/lib")
# 添加交叉编译需要使用的动态库目录

set(TARGET_NAME template)#project name
set(SOURCE_FILES main.cpp)
add_executable(${TARGET_NAME} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} sail)
target_link_libraries(${TARGET_NAME} opencv_core)   # 链接 OpenCV 核心模块
target_link_libraries(${TARGET_NAME} opencv_imgcodecs)  # 链接图像编解码模块
target_link_libraries(${TARGET_NAME} opencv_imgproc)  # 链接图像处理模块
# sail为需要链接的库