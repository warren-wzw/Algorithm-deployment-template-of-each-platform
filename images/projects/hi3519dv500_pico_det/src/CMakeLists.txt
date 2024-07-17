# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
#debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
# project information
project(ACL_CLASSIFICATION_RESNET50)

# Compile options
add_compile_options(
    -std=c++11
    -fPIC
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie")
set(CMAKE_SKIP_RPATH TRUE)

# Header path
include_directories(
$ENV{DDK_PATH}/acllib/include/
    ../inc/
    ../3rdparty/opencv/opencv-3.4.3/lib
)

# add host lib path
if (target MATCHES "Simulator_Function" OR target MATCHES "Simulator_Instruction" OR target MATCHES "Simulator_Performance")
    set(ENV{NPU_HOST_LIB} $ENV{DDK_PATH}/toolkit/tools/sim/lib)
else()
    set(ENV{NPU_HOST_LIB} $ENV{DDK_PATH}/acllib/lib64/stub) #this case
endif()

# opencv
set(OpenCV_DIR "../3rdparty/opencv/opencv-3.4.3/share/OpenCV")
# 查找opencv的path
find_package(OpenCV REQUIRED)
# 打印opencv的相关信息
message(STATUS "OpenCV library status:")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
#
set(LIB_PATH $ENV{NPU_HOST_LIB} )
MESSAGE(NPU_HOST_LIB=$ENV{NPU_HOST_LIB})
link_directories(${LIB_PATH})
set(OPENCV_LIB ${OpenCV_DIR})
link_directories(${OPENCV_LIB})
message(${OPENCV_LIB})
FILE(GLOB SAMPLE_SRC_FILE *.cpp)

add_executable(main ${SAMPLE_SRC_FILE})

if (target MATCHES "Simulator_Function")
    target_link_libraries(main funcsim  )
elseif (target MATCHES "Simulator_Instruction" OR target MATCHES "Simulator_Performance")
    target_link_libraries(main instsim  )
else()
    target_link_libraries(main svp_acl ss_mpi_sysmem ss_mpi_km pthread securec dl protobuf-c stdc++ ${OpenCV_LIBS} m)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
