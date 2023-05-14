#! /bin/bash

## Multinode config
GPUS_PER_NODE=4
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
NNODES=$NHOSTS
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo WORLD_SIZE $WORLD_SIZE
HOSTFILE_NAME=./hostfile_${JOB_ID}
cat $SGE_JOB_HOSTLIST > $HOSTFILE_NAME

# Data path
DATA_PATH='
       1 /bb/grandchallenge/gaf51090/datasets/aozora_books/binarized/sp_identity/aozora_books_text_document
       5 /bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/sp_identity/en_wiki_text_document
       5 /bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/sp_identity/ja_wiki_text_document
       12.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc_filtered_org-bwords00_text_document
       12.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc_filtered_org-bwords01_text_document
       12.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc_filtered_org-bwords02_text_document
       12.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc_filtered_org-bwords03_text_document
       9.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc2_filtered_org-bwords00_text_document
       9.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc2_filtered_org-bwords01_text_document
       9.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc2_filtered_org-bwords02_text_document
       9.125 /bb/grandchallenge/gaf51090/datasets/cacc/binarized/sp_identity/ca_cc2_filtered_org-bwords03_text_document
       1 /bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sp_identity/redpajama_github00_text_document
       1 /bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sp_identity/redpajama_github01_text_document
       1 /bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sp_identity/redpajama_github02_text_document
       1 /bb/grandchallenge/gaf51090/datasets/redpajama_github/binarized/sp_identity/redpajama_github03_text_document
'


# Model size
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16

# Other experimental parameters
EXP_NAME=pretrain_gpt_345m_64gpu_sp_identity_debug_training
CHECKPOINT_PATH=/bb/grandchallenge/gaf51090/checkpoints/${EXP_NAME}
TENSORBOARD_PATH=/bb/grandchallenge/gaf51090/logs/${EXP_NAME}
WANDB_NAME=${EXP_NAME}
export WANDB_DIR='/bb/grandchallenge/gaf51090/wandb'
VOCAB_FILE='/bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model'

mpirun -np $WORLD_SIZE -npernode $GPUS_PER_NODE --hostfile $HOSTFILE_NAME python pretrain_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTENTION_HEADS \
       --micro-batch-size 4 \
       --global-batch-size 1280 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --tensorboard-dir $TENSORBOARD_PATH \
       --wandb-name $WANDB_NAME \
       --train-samples 64000000 \
       --data-path $DATA_PATH \
       --tokenizer-type JapaneseSentencePiece \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --rampup-batch-size 256 256 50000 \
       --init-method-std 0.018 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --weight-decay 0.1 \
       --log-interval 1 \
       --save-interval 500 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16 \
       --checkpoint-activations \
       --distribute-checkpointed-activations \
       --no-scatter-gather-tensors-in-pipeline \
       --use-mpi \
       --log-timers-to-tensorboard

rm $HOSTFILE_NAME

