#!/bin/bash
model_dir=$(dirname $(readlink -f "$0"))

if [ ! $1 ]; then
    target=bm1684x
else
    target=$1
fi

outdir=../models/BM1684X

function gen_mlir()
{
    model_transform.py \
        --model_name test_output \
        --model_def ../models/onnx/model.onnx \
        --mlir test_output_$1b.mlir
}

function gen_fp32bmodel()
{
    model_deploy.py \
        --mlir test_output_$1b.mlir \
        --quantize F32 \
        --chip $target \
        --model test_output_fp32_$1b.bmodel

    mv test_output_fp32_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_fp32bmodel 1

popd
