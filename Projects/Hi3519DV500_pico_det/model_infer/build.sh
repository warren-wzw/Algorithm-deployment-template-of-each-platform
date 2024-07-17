#!/bin/bash
# Copyright (C) Shenshu Technologies Co., Ltd. 2023-2023. All rights reserved.
set -e

CUR_DIR=$(cd $(dirname $0) && pwd -P)

function prepare_out_env()
{
    if [ -d "out" ]; then
        rm -rf out
    fi
}

function prepare_env()
{
    if [ -d "build" ]; then
        rm -rf build
    fi
    mkdir -p build/intermediates
}


function set_build_env()
{
    if [ -z "$DDK_PATH" ]; then
        echo "ERROR: env DDK_PATH is empty, please set DDK_PATH"
        exit
    else
        echo "DDK_PATH=$DDK_PATH"
    fi
    export LANG="C"
}

function build_acl()
{
    target_type=Simulator_Function
    build_type=Release
    tool_chain=g++

    until [ $# -eq 0 ]
    do
        if [ "$1"x = "func"x ]; then
            echo "build target is Simulator_Function"
            target_type="Simulator_Function"
        elif [ "$1"x = "inst"x ]; then
            echo "build target is Instruction_Function"
            target_type="Simulator_Instruction"
        elif [ "$1"x = "release"x ]; then
            echo "build project in Release mode"
            build_type="Release"
        elif [ "$1"x = "debug"x ]; then
            echo "build project in Debug mode"
            build_type="Debug"
        elif [ "$1"x = "g++"x ]; then
            echo "tool_chain is g++"
            tool_chain=g++
        elif [ "$1"x = "board"x ]; then
            echo "build target is board"
            echo "toolchain is arm"
            target_type="board"
            tool_chain=aarch64-v01c01-linux-musl-gcc
        elif [ "$1"x = "musl"x ]; then
            echo "toolchain is aarch64-v01c01-linux-musl-gcc"
            target_type="board"
            tool_chain=aarch64-v01c01-linux-gnu-gcc
        elif [ "$1"x = "glibc"x ]; then
            echo "toolchain is aarch64-v01c01-linux-gnu-gcc"
            target_type="board"
            tool_chain=aarch64-v01c01-linux-gnu-gcc
        else
            echo "unknow option $1"
            exit 1
        fi
        shift
    done

    cd build/intermediates
    cmake ../.. -Dtarget=$target_type -DCMAKE_BUILD_TYPE=$build_type -DCMAKE_CXX_COMPILER=$tool_chain
    make clean && make -j32
    cd -    
}


prepare_out_env
set_build_env

prepare_env
build_acl func
mv out/main out/func_main
prepare_env
build_acl inst
mv out/main out/inst_main

SWITCH_CROSS_LIB_DIR=$DDK_PATH/acllib/script
chmod +x $SWITCH_CROSS_LIB_DIR/switch_cross_lib.sh

prepare_env
cd $SWITCH_CROSS_LIB_DIR
bash $SWITCH_CROSS_LIB_DIR/switch_cross_lib.sh aarch64-v01c01-linux-musl
cd $CUR_DIR
build_acl board musl
mv out/main out/board_musl_main

prepare_env
cd $SWITCH_CROSS_LIB_DIR
bash $SWITCH_CROSS_LIB_DIR/switch_cross_lib.sh aarch64-v01c01-linux-gnu
cd $CUR_DIR
build_acl board glibc
    mv out/main out/board_glibc_main
