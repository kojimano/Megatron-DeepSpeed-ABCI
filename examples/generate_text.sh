#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
MEGATRON_DEEPSPEED_PATH=/home/deepspeed/repo/Megatron-DeepSpeed
CHECKPOINT_PATH=$MEGATRON_DEEPSPEED_PATH/dataset/checkpoints/gpt2_345m
VOCAB_FILE=$MEGATRON_DEEPSPEED_PATH/dataset/gpt2-vocab.json
MERGE_FILE=$MEGATRON_DEEPSPEED_PATH/dataset/gpt2-merges.txt
#b=8
b=1
mp=1
experts=1
nodes=1
gpus=1


use_tutel=""
#use_tutel="--use-tutel"


ds_inference=""
#ds_inference="--ds-inference"

launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus"
L=24
H=1024
#H=2048
A=16
#experts1=${experts[$k]}
#program_cmd="tools/gen.py \
program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --pipeline-model-parallel-size 1  \
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
       --out-seq-length 50 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --num-samples 0 \
       --load $CHECKPOINT_PATH \
       $use_tutel $ds_inference"
       #--load ./dataset/checkpoints/gpt2_345m \
       #--no-masked-softmax-fusion \
       #--num-samples 1 \
       #--seq-length 1024 \
       #--out-seq-length 1024 \

# TODO: Try w/o load checkpoint, see if it takes a different code path
# TODO: MoE sub-class, MegatronGPT

echo $launch_cmd $program_cmd
#sudo $launch_cmd $program_cmd
$launch_cmd $program_cmd
