# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
# project information
project(PICO_DET)

add_subdirectory("./src")
