#!/bin/bash

#$ -l rt_F=8
#$ -l h_rt=0:20:00
#$ -j y
#$ -cwd
#$ -o abci/jobs/results/o.$JOB_ID

source /etc/profile.d/modules.sh
module load python/3.10/3.10.10 cuda/11.8/11.8.0 cudnn/8.6/8.6.0 nccl/2.16/2.16.2-1 hpcx/2.12
cd /home/acf15317dw/Megatron-DeepSpeed-ABCI
source /home/acf15317dw/megatron-deepspeed/bin/activate

sh abci/shells/pretrain_gpt_13b_mp4_pp8.sh
