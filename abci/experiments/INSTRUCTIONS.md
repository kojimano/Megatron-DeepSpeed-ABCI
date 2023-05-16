# ABCI Grand Challenge Instructions
## Overview
### Model
The training is carried out on dense Transformers consisting of 13B parameters. The table below provides a summary of the model hyperparameters.

| Hyperparameter | Value |
|----------|----------|
| Hidden Size | 5120|
| Number of Attention Heads |40|
| Number of Layers |40|
| Sequence Length |2048|

### Data
We're targeting 139.2B tokens for training. The data breakdown is as follows. A Sentencepiece-based Tokenizer is used on Japanese/English Wikipedia, Japanese Common Crawl, and Code. No normalization is applied to the tokenizer. You can find more details [here](https://github.com/kojimano/Megatron-DeepSpeed-ABCI/blob/main/abci/tokenizer/README.md).

| Dataset | # Tokens | Language |
|----------|----------|----------|
| Aozora Books   | 1.3B | Japanese  |
| RedPajama Code   | 5.5B  | Code  |
| Japanese Wikipedia   | 6.9B  | Japanese  |
| English Wikipedia   | 6.9B  | English  |
| Japanese Common Crawl | 118.3B  | Japanese  |
| **Total**   | **139.2B**   | **-**   | 

## Instructions
The experiment duration is from May 16, 2023 (Tuesday) 11 am JST to May 23, 2023 (Tuesday) 11 am JST. In EST, this is from May 15, 2023 (Monday) 10 pm to May 22, 2023 (Monday) 10 pm.

### 1. Launching Jobs
Initiate the experiment using 544 GPUs with the following command:
```bash
cd /bb/grandchallenge/gaf51090/Megatron-DeepSpeed-ABCI
qsub -ar 23682 -g gaf51090 ./abci/jobs/submit_pretrain_gpt_13b_544gpu_commonspace_start.sh
```
All scripts are located under `/bb/grandchallenge/gaf51090/Megatron-DeepSpeed-ABCI` and the Python virtual environment is located at `/bb/grandchallenge/gaf51090/megatron-deepspeed`.

### 2. Monitoring Jobs
- Monitor the progress of training on the [WandB](https://wandb.ai/gpt-fugaku/gpt-abci?workspace=user-kojimano) logging board.
- Alternatively, Tensorboard logging is available at `/bb/grandchallenge/gaf51090/logs`.

### 3. Troubleshooting

#### 3.1 If Training Loss Diverges
1. Continue training for an additional 1-2 hours to check if it recovers.
2. If the problem persists, stop the training process. First, obtain the job id with `qstat -u acf15317dw`, then execute `qdel $JOBID`.
3. To resume training, run the model from the last known good checkpoints and change the random seed:
  - Change `--seed` in `./abci/shells/pretrain_gpt_13b_544gpu_commonspace_resume.sh`.
  - Identify the checkpoints previous to the divergence of training loss by running  `ls /bb/grandchallenge/gaf51090/checkpoints/13b_544gpu_commonspace_start_tr2`.
  - Update the file  `/bb/grandchallenge/gaf51090/checkpoints/13b_544gpu_commonspace_start_tr2/latest_checkpointed_iteration.txt` with the chosen checkpoints.
  - Restart the job with `qsub -ar 23682 -g gaf51090 ./abci/jobs/submit_pretrain_gpt_13b_544gpu_commonspace_resume.sh`.
  - The training should resume and log new entries in WandB.

#### 3.2 If Training Process Dies
1. Try to resume the training by simply submitting one of the following commands: 
```bash
qsub -ar 23682 -g gaf51090 ./abci/jobs/pretrain_gpt_13b_544gpu_commonspace_start.sh
```
or
```bash
qsub -ar 23682 -g gaf51090 ./abci/jobs/pretrain_gpt_13b_544gpu_commonspace_resume.sh
```
2. The training should resume and new entries should be recorded in the WandB log.
3. If the above steps don't resolve the issue, there might be a problem with the GPUs. Reach out to Noriyuki Kojima or Shunkai Nakamura for further investigation. Try taking a look at job log by locating a log `ls /bb/grandchallenge/gaf51090/job_outputs/submit_pretrain_gpt_13b_544gpu_commonspace_start*` and taking it a look.
5. In case of broken GPUs, the `--global-batch-size` parameter must be adjusted to be a multiple of `DP * 2`, likely `--global-batch-size 1280`. Then, proceed with resubmitting the job using one of the commands mentioned in step 1. Use `pretrain_gpt_13b_544gpu_commonspace_resume.sh` if you had previously experienced training divergence.

