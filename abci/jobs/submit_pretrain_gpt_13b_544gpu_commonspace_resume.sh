#!/bin/bash
#$ -l rt_F=136
#$ -l h_rt=162:00:00
#$ -j y
#$ -cwd
#$ -o /bb/grandchallenge/gaf51090/job_outputs/13b_544gpu_commonspace_start_tr2.$JOB_ID.out
#$ -e /bb/grandchallenge/gaf51090/job_outputs/13b_544gpu_commonspace_start_tr2.$JOB_ID.err

source /etc/profile.d/modules.sh
module load python/3.10/3.10.10 cuda/11.8/11.8.0 cudnn/8.6/8.6.0 nccl/2.16/2.16.2-1
module unload intel-mpi/2021.8
module load hpcx/2.12
source /bb/grandchallenge/gaf51090/megatron-deepspeed/bin/activate
cd /bb/grandchallenge/gaf51090/Megatron-DeepSpeed-ABCI
sh abci/shells/pretrain_gpt_13b_544gpu_commonspace_resume.sh
