#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
#CHECKPOINT_PATH=./dataset/checkpoints/gpt2_345m
CHECKPOINT_PATH=/home/deepspeed/blobstore/users/conglli/checkpoint/moe/gpt3-350m-ds-moe-fixed-dropoutfix-mp-1-ep-128-gpus-128-mlc-0.01-lr-2.0e-4-bs-256/
VOCAB_FILE=./dataset/gpt2-vocab.json
MERGE_FILE=./dataset/gpt2-merges.txt
b=128
mp=1
experts=128
nodes=1
gpus=2


use_tutel=""
#use_tutel="--use-tutel"


ds_inference=""
#ds_inference="--ds-inference"

launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus"
L=24
H=1024
A=16
#experts1=${experts[$k]}
program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers $L \
       --hidden-size $H \
       --num-attention-heads $A \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --num-experts ${experts} \
       --mlp-type standard \
       --micro-batch-size $b \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples 0 \
       --load $CHECKPOINT_PATH \
       --deepspeed \
       --deepspeed_config /home/deepspeed/repo/Megatron-DeepSpeed/examples/compression/ds_config_gpt_TEMPLATE.json \
       $use_tutel $ds_inference"
       #--finetune \

echo $launch_cmd $program_cmd
$launch_cmd $program_cmd
