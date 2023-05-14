#!/bin/bash
#$ -l rt_F=32
#$ -l h_rt=0:20:00
#$ -j y
#$ -cwd
#$ -o /bb/grandchallenge/gaf51090/job_outputs/pretrain_gpt_13b_128gpu_debug_abci.$JOB_ID

source /etc/profile.d/modules.sh
module load python/3.10/3.10.10 cuda/11.8/11.8.0 cudnn/8.6/8.6.0 nccl/2.16/2.16.2-1
module unload intel-mpi/2021.8
module load hpcx/2.12
source /home/acf15317dw/megatron-deepspeed/bin/activate
cd /home/acf15317dw/Megatron-DeepSpeed-ABCI
sh abci/shells/pretrain_gpt_13b_128gpu_debug_abci.sh
