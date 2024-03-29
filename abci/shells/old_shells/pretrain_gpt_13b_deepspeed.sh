#! /bin/bash
# Runs the "13B" parameter model
GPUS_PER_NODE=4
## Change for multinode config
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
NNODES=$NHOSTS
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo WORLD_SIZE $WORLD_SIZE

DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/13b_mp4_pp8_2/${JOB_ID}/
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt

HOSTFILE_NAME=./hostfile_${JOB_ID}
cat $SGE_JOB_HOSTLIST > $HOSTFILE_NAME


# 13 B Deepspeed Stage1
mpirun -np $WORLD_SIZE -npernode $GPUS_PER_NODE --hostfile $HOSTFILE_NAME python pretrain_gpt.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 2 \
       --num-layers 10 \
       --hidden-size 5120 \
       --num-attention-heads 40 \
       --micro-batch-size 1 \
       --global-batch-size 88 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500 \
       --lr-decay-iters 50 \
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
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 250 \
       --eval-iters 10 \
       --deepspeed_config examples/deepspeed_config/ds_zero_stage_2.config \
       --fp16 \
       --checkpoint-activations \
       --use-mpi


rm $HOSTFILE_NAME
