cmake_minimum_required(VERSION 3.4.1)
set(PROJECT_NAME rknn_template)
project(${PROJECT_NAME})

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(STATUS "64bit")
	set(LIB_ARCH lib64)
else()
	message(STATUS "32bit")
	set(LIB_ARCH lib)
endif()

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

# opencv
if(LIB_ARCH STREQUAL "lib")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-3.4.3/share/OpenCV)
    message(STATUS "--------armhf")
else()
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
    message(STATUS "aarch64")
endif()
find_package(OpenCV REQUIRED)

#stb
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty/)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(${PROJECT_NAME}
        src/main.cc
)

target_link_libraries(${PROJECT_NAME}
        ${RKNN_API_LIB}
        ${OpenCV_LIBS}
		dl
        )

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)
