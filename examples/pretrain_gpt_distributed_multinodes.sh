#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
## Change for multinode config
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
NNODES=$NHOSTS
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo WORLD_SIZE $WORLD_SIZE

DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2/${JOB_ID}/
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt

HOSTFILE_NAME=./hostfile_${JOB_ID}
cat $SGE_JOB_HOSTLIST > $HOSTFILE_NAME

mpirun -np $WORLD_SIZE -npernode $GPUS_PER_NODE --hostfile $HOSTFILE_NAME python \
       pretrain_gpt.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 32 \
       --global-batch-size 512 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 5000 \
       --lr-decay-iters 3200 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --use-mpi

rm $HOSTFILE_NAME
