#!/bin/bash

cur_dir=`cd $(dirname $0) && pwd`
work_dir=`cd $cur_dir/../.. && pwd`
data_dir=$work_dir/data
res_dir=$work_dir/results
cache_dir=$res_dir/cache/
code_dir=$work_dir/Code/Edit
hparams_dir=$work_dir/hparams
plm_dir=$work_dir/../../pretrained_models/

plm=llama-2-7b
dataset=counterfact
method=DISCO

signature=${plm}/${dataset}/${method}
ckpt_dir=$res_dir/${signature}_k${k}_w${w}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=$cache_dir
export TORCH_EXTENSIONS_DIR=$cache_dir/torch_extension/
export OMP_NUM_THREADS=20
export TF_ENABLE_ONEDNN_OPTS=0
export TRANSFORMERS_CACHE=$cache_dir
export MPLCONFIGDIR=$cache_dir

rm -rf $ckpt_dir
if [ ! -d $ckpt_dir ]; then
    mkdir -p $ckpt_dir
    chmod -R 777 $ckpt_dir
fi

k=1

python=/opt/anaconda3/bin/python3
$python $code_dir/run_counter.py \
    --editing_method $method \
    --alpha $w \
    --hparams_dir $hparams_dir/$method/${plm}_$k.yaml \
    --train_file $data_dir/counterfact/counterfact-train.json \
    --data_file $data_dir/portability/one_hop/counterfact_portability_gpt4.json \
    --save_dir $ckpt_dir | tee -a $ckpt_dir/run.log