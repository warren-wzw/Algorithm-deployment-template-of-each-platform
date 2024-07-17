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

function gen_cali_table()
{
    run_calibration.py test_output_$1b.mlir \
        --dataset ../datasets \
        --input_num 100 \
        -o test_cali_table
}

function gen_int8bmodel()
{
    model_deploy.py \
        --mlir test_output_$1b.mlir \
        --quantize INT8 \
        --chip $target \
        --calibration_table test_cali_table \
        --asymmetric \
        --model test_output_int8_$1b.bmodel

    mv test_output_int8_$1b.bmodel $outdir/
}

pushd $model_dir
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi
# batch_size=1
gen_mlir 1
gen_cali_table 1
gen_int8bmodel 1

# batch_size=4
gen_mlir 4
gen_int8bmodel 4

popd