#!/bin/bash

cur_dir=`cd $(dirname $0) && pwd`
work_dir=`cd $cur_dir/../.. && pwd`
data_dir=$work_dir/data
res_dir=$work_dir/results
cache_dir=$res_dir/cache/
code_dir=$work_dir/Code/Edit
hparams_dir=$work_dir/hparams
plm_dir=$work_dir/../../pretrained_models/

plm=gpt-j-6b
dataset=zsre
method=FT

signature=${plm}/${dataset}/${method}
ckpt_dir=$res_dir/$signature

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

python=/opt/anaconda3/bin/python3
$python $code_dir/run_zsre.py \
    --editing_method $method \
    --hparams_dir $hparams_dir/$method/$plm.yaml \
    --data_file $data_dir/portability/one_hop/zsre_mend_eval_portability_gpt4.json \
    --save_dir $ckpt_dir | tee -a $ckpt_dir/run.log

    # --max_steps 1500 \
    # --save_total_limit 3 \
#  -m torch.distributed.launch --master_port 25601 --nproc_per_node=4 \
#     $code_dir/trainer_vanilla/run_clm_llms.py \
#     --model_name_or_path $plm_dir/$pretrained_model \
#     --train_file $train_data --do_train \
#     --output_dir $ckpt_dir \
#     --streaming \
#     --preprocessing_num_workers 16 \
#     --keep_linebreaks False \
#     --ignore_data_skip True \
#     --logging_steps 1 \
#     --save_steps 200 \
#     --save_strategy steps \
#     --num_train_epochs 3 \
#     --learning_rate 2e-5 \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --block_size 768 \
#     --lr_scheduler_type "cosine" \
#     --deepspeed $plm_dir/ds_config.json \
#     --only_optimize_layers "29" "28" "27" "26" "25" "24" "23" "22" "21" "20" "19" "18" "17" "16" "15" "14" "13" "12" "11" "10" "9" "8" "7" "6" "5" "4" "3" "2" "1" "0" \
#     --gradient_accumulation_steps 64 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --overwrite_cache \
#     --cache_dir $cache_dir \
#     --seed 42 \
#     --fp16 \
#     $overwrite_args \
# 2>&1 | tee -a $log_file
#     # --resume_from_checkpoint $ckpt_dir/checkpoint-685 \

# bash $work_dir/infer/run_infer.sh wmt22 alpaca_infer $signature last 1 4
# bash $work_dir/inference/run_infer.sh wmt22 mt_infer $signature 1600 1 4
# bash $work_dir/inference/run_infer.sh wmt22 mt_infer $signature 1400 1 4
# bash $work_dir/inference/run_infer.sh wmt22 mt_infer $signature 1200 1 4
# bash $work_dir/inference/run_infer.sh wmt22 mt_infer $signature 1000 1 4