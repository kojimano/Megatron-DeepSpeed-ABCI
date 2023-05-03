#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd
#$ -o abci/jobs/results/o.$JOB_ID

source /etc/profile.d/modules.sh
module load python/3.10/3.10.10 cuda/11.8/11.8.0 cudnn/8.6/8.6.0 nccl/2.16/2.16.2-1
source /home/acd13570uk/megatron-deepspeed/bin/activate

sh sh abci/shells/pretrain_gpt_abci.sh

