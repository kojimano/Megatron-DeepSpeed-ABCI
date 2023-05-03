### megatron-deepspeed-abciの環境構築手順

#### Prepare dataset and vocab
```bash
git clone git@github.com:kojimano/Megatron-DeepSpeed-ABCI.git
cd Megatron-DeepSpeed-ABCI
cd dataset
sh download_books.sh
sh download_vocab.sh
cd ..
```

#### Prepare Python environment

```bash
module load python/3.10/3.10.10
python3 -m venv megatron-deepspeed
source megatron-deepspeed/bin/activate
```

#### Install packages
```bash
pip install -r requirements.txt 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed
```

#### Set-up GPU packages
```bash
# launch GPU interactive job
qrsh -g gaf51090 -l rt_F=8 -l h_rt=1:00:00 
qsub -g gaf51090 submit_pretrain_gpt_distributed_multinodes_with_mp_abci.sh
qstat

# launch module
module load cuda/11.8/11.8.0 cudnn/8.6/8.6.0 nccl/2.16/2.16.2-1
module load hpcx/2.12
module load intel-mpi/2021.8
module list
> Currently Loaded Modulefiles:
 1) python/3.10/3.10.10   2) cuda/11.8/11.8.0   3) cudnn/8.6/8.6.0   4) nccl/2.16/2.16.2-1

# install Apex
git clone https://github.com/NVIDIA/apex
cd apex
source megatron-deepspeed/bin/activate
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### Toy experiments
```bash
#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

rm -rf checkpoints/book_debug
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/book_debug
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt
LOCAL_RANK=0 

python pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 4 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 50 \
       --lr-decay-iters 32 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16


rm -rf checkpoints/book_debug
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/book_debug
VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt
LOCAL_RANK=0 

GPUS_PER_NODE=4
# Change for multinode config
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --global-batch-size 32 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
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
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16

```
