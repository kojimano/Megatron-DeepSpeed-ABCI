# Instructions for ABCI grand-challenge
## Overview
### Model
- We will train 13B parameters model. 
- The model is a dense Transfromer with the maximimum sequence length of 2048.
- This Transformer has a 

### Data
- We will train models for 150B tokens
- All data are processed using our custom sentencepiece tokenizer trained on Japanese and English wikipedia as well as code data.
- The aggregated dataset is summarized below.

| Dataset | # Tokens |  Weight | # Final Tokens | Language |
|----------|----------|----------|----------|----------|
| Aozora Books   | X   | 3   | 0.5B   | Japanese  |
| Japanese Wikipedia   | X   | 5   | 5B  |  Japanese   |
| CACC   | X   | 0.5   | 135B   | Japanese  |
| Abema Blog   | X   | 10   | 0.1B   | Japanese  |
| English Wikipedia   | X   | 5    | 5B   | English  |
| RedPajama Code   | X   | 0.1   | 5B   | Code  |
| Total   | -   | -   | 150.6B     | -  |

## Instructons
The duration of experiment is from 2023/05/16(Tue) 11 am JST to 2023/05/23(Tue) 11 am JST.
This is from 2023/05/15(Mon) 10 pm EST to 2023/05/23(Mon) 10 pm EST.

### 1. Launch Jobs
Launch the experiment with 544 GPUs.
```bash
cd /home/acf15317dw/Megatron-DeepSpeed-ABCI
qsub -ar 23682 -g gaf51090 ./abci/jobs/submit_pretrain_gpt_13b_544gpu_sp_identity_debug_abci_ver3.sh
```

### 2. Monitor Jobs
Check job logging at `hello` and tensorboard at `hello`.

### 3. Troubleshooting (TODO)
#### 3.1 Training dies
- Check error logging at `hello` to identify the source of error.
- Investigate any faulty GPUs by running `hello`
- Remove the faulty GPU nodes from `hostfle` (you might also need to adjust $WORLD_SIZE)
- Resume experiments by re-submitting `qsub -ar 23682 -g gaf51090 ./abci/jobs/13b_544gpus.sh`

#### 3.2 Training diverges

#### 3.3 Stucking after  `compiling and loading fused kernels`
- Delete `megatron/fused_kernels` and rerun scripts.


### 4. compiling and loading fused kernels
