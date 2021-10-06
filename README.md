# Megatron-DeepSpeed

Megatron-DeepSpeed is a repository for training transformer-based language models using an integration of [Megatron](https://github.com/NVIDIA/Megatron-LM) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). This integration brings many of DeepSpeed's latest features to Megatron's training infrastructure.

Combined, this repository can train language models with **tens of trillions** of parameters.

# Contents
   * [Contents](#contents)
   * [DeepSpeed Feature Overview](#deepspeed-features)
   * [Setup](#setup)
   * [Usage](#usage)
# DeepSpeed Feature Overview

* 3D parallelism
* ZeRO-powered data parallelism
  * CPU and NVMe offloading

**TODO:** table mapping features to Megatron training paths (e.g., BERT/GPT/T5)?


# Setup

## Install Requirements
This repository requires PyTorch, CUDA, NCCL, and DeepSpeed for distributed GPU training.

Note: PyTorch 1.8+ is strongly encouraged for best training performance.

Install dependencies via:
```
pip install -r requirements.txt
```


After installing the required packages, we recommend running `ds_report` to query
the DeepSpeed installation environment:
```
$ ds_report
```
### Install Apex

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .
```

# Usage
