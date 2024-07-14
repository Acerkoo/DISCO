#!/bin/bash

cur_dir=`cd $(dirname $0) && pwd`
work_dir=`cd $cur_dir/.. && pwd`
data_dir=$work_dir/data
res_dir=$work_dir/results
cache_dir=$res_dir/cache/
code_dir=$work_dir/Edit
eval_dir=$work_dir/eval
hparams_dir=$code_dir/hparams
plm_dir=$work_dir/../../pretrained_models/

# plm=llama-2-7b
plm=$1
dataset=$2
# portability
method=$3
# MEMIT
# bak=$4

signature=${plm}/${dataset}/${method}
ckpt_dir=$res_dir/$signature

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$cache_dir
export TORCH_EXTENSIONS_DIR=$cache_dir/torch_extension/
export OMP_NUM_THREADS=20
export TF_ENABLE_ONEDNN_OPTS=0
export TRANSFORMERS_CACHE=$cache_dir
export MPLCONFIGDIR=$cache_dir

# rm -rf $ckpt_dir
# if [ ! -d $ckpt_dir ]; then
#     mkdir -p $ckpt_dir
#     chmod -R 777 $ckpt_dir
# fi

# python=/opt/anaconda3/bin/python3
python $eval_dir/evaluate.py \
    --tokenizer_path $plm_dir/$plm \
    --res_dir $ckpt_dir 
