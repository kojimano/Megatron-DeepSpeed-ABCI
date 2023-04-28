#!/bin/bash

#$ -l rt_F=8
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd
#$ -o results/o.$JOB_ID

source /etc/profile.d/modules.sh
module load python/3.10/3.10.10 cuda/11.8/11.8.0 cudnn/8.6/8.6.0 nccl/2.16/2.16.2-1 hpcx/2.12
source /home/acd13570uk/megatron-deepspeed/bin/activate


sh examples/pretrain_gpt_distributed_multinodes.sh
