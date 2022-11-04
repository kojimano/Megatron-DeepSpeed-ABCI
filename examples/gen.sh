#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.6+PTX
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
b=8
mp=1
nodes=1
gpus=1

launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus"
L=24
H=1024
A=16

program_cmd="tools/gen.py \
       --no-masked-softmax-fusion \
       --tensor-model-parallel-size $mp \
       --num-layers $L \
       --hidden-size $H \
       --num-attention-heads $A \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --micro-batch-size $b \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --top_p 0.9 \
       --seed 42 \
       --num-samples 2 \
       --genfile out.json \
       --load ${CHECKPOINT_PATH}"

echo $launch_cmd $program_cmd
sudo $launch_cmd $program_cmd
