# Wikipedia Data Preparation

This README describes the process of downloading, cleaning, formatting, tokenizing, and binarizing Wikipedia data in Japanese and English.

## Table of Contents
- [Procedures](#procedures)
  * [1. Get computes (ABCI) and install packages](#1-get-computes-abci-and-install-packages)
  * [2. Download, clean and format the latest Wikipedia (ja, en) dumps](#2-download-clean-and-format-the-latest-wikipedia-ja-en-dumps)
  * [3. Tokenize and binarize data into the megatron-deepspeed format](#3-tokenize-and-binarize-data-into-the-megatron-deepspeed-format)
- [Dataset Statistics](#dataset-statistics)
  * [Basic Statistics](#basic-statistics)
  * [Data Paths](#data-paths)

## Procedures

### 1. Get computes (ABCI) and install packages

```bash
# 1.1 get computes
qrsh -g gaf51090 -l rt_C.small=1 -l h_rt=2:00:00 

# 1.2 activate your virtual environment
source XXX/bin/activate

# 1.3 install wikiextractor
pip install wikiextractor
```

### 2. Download, clean and format the latest Wikipedia (ja, en) dumps

```bash
# 2.1 generate data files with a loose json (jsonl) format (each file is 100MB)
python -m abci.dataset.wikipedia.wikidump_download
```

### 3. Tokenize and binarize data into the megatron-deepspeed format

```bash
# 3.1 merge smaller jsonl files into a single jsonl

# 3.2 tokenize and binarize data
```

## Dataset Statistics

### Basic Statistics

- Processing time calculated using `rt_C.small=1`

| Language | # Extracted Articles | Merged Jsonl Size | # Tokens (GPT-2) | # Tokens (Rinna) | Processing Times (1.1/2.1/2.2) |
| -------- | -------------------- | ----------------- | --------------- | --------------- | ----------------------------- |
| Japanese | 2,219,610            | 6.9 GB            | -               | -               | 38 mins /                           |
| English  | -                    | -                 | -               | -               | -                             |

### Data Paths

- Paths under `/bb/grandchallenge/gaf51090/datasets`

| Language | Compressed Raw Data       | Processed jsonl files (after step 2) | Merged jsonl (after step 3.1) | Binarized Data (GPT-2) | Binarized Data (Rinna) |
| -------- | ------------------------- | ------------------------------------- | --------------------------- | --------------------- | --------------------- |
| Japanese | wikipedia/raw_data/ja/ja_xml.bz2 | wikipedia/processed/ja/AA            | -                           | -                     | -                     |
| English  | wikipedia/raw_data/en/en_xml.bz2 | wikipedia/processed/en/AA            | -                           | -                     | -                     |
