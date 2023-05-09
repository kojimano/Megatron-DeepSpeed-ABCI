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
# generate data files with a loose json (jsonl) format (each file is 100MB)
python -m abci.dataset.wikipedia.wikidump_download
```

## 3. Tokenize and binarize data into the megatron-deepspeed format

To tokenize and binarize the data, follow these steps:

### 3.1. Merge smaller jsonl files into a single jsonl:

   **Japanese:**

   ```
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/processed/ja/AA /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja ja_merged
   ```

   **English:**

   ```
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/processed/en/AA /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en en_merged_1
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/processed/en/AB /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en en_merged_2
   ./abci/dataset/wikipedia/merge_files.sh /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en en_merged
   ```

### 3.2. Tokenize and binarize data:

   **GPT-2:**

   Set the output directory:

   ```
   export OUTDIR=/bb/grandchallenge/gaf51090/datasets/wikipedia/binarized/gpt-2
   mkdir -p $OUTDIR
   ```

   Tokenize and binarize **Japanese debug (100MB file)**:

   ```
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/processed/ja/AA/wiki_00 \
          --output-prefix $OUTDIR/ja_wiki_100mb \
          --vocab dataset/gpt2-vocab.json \
          --merge-file dataset/gpt2-merges.txt \
          --dataset-impl mmap \
          --tokenizer-type GPT2BPETokenizer \
          --workers 64 \ # login-node
          --append-eod
   ```

   Tokenize and binarize **Japanese**:

   ```
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja/ja_merged.json \
          --output-prefix $OUTDIR/ja_wiki \
          --vocab dataset/gpt2-vocab.json \
          --merge-file dataset/gpt2-merges.txt \
          --dataset-impl mmap \
          --tokenizer-type GPT2BPETokenizer \
          --workers 16 \
          --append-eod
   ```

   Tokenize and binarize **English**:

   ```
   python tools/preprocess_data.py \
          --input /bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en/en_merged.json \
          --output-prefix $OUTDIR/en_wiki \
          --vocab dataset/gpt2-vocab.json \
          --merge-file dataset/gpt2-merges.txt \
          --dataset-impl mmap \
          --tokenizer-type GPT2BPETokenizer \
          --workers 64 \
          --append-eod
   ```

## Dataset Statistics

### Basic Statistics

- Processing time calculated using `rt_C.small=1`
- (†) uses `rt_C.large=1`


| Language | # Extracted Articles | Merged Jsonl Size |  # Tokens / # Documents (GPT-2) | # Tokens / # Documents (Rinna) | Processing Times (2/3.1/3.2) |
| -------- | -------------------- | ----------------- | --------------- | --------------- | ----------------------------- |
| Japanese | 2,219,610            | 6.9 GB            | 1,802,750,651 / 2,219,610       | -               | 38 mins / 1 <mins / 70<? mins |
| English  | 17,020,965           | 17.4 GB           | 3,517,216,353 / 17,020,965        | -               | 208 mins / 1 <mins     / 15† mins   |

### Data Paths

- Paths under `/bb/grandchallenge/gaf51090/datasets/wikipedia`

| Language | Compressed Raw Data       | Processed jsonl files (after step 2) | Merged jsonl (after step 3.1) | Binarized Data (GPT-2) | Binarized Data (Rinna) |
| -------- | ------------------------- | ------------------------------------- | --------------------------- | --------------------- | --------------------- |
| Japanese | raw_data/ja/ja_xml.bz2 | processed/ja/AA            | merged/ja/ja_merged.json   | binarized/gpt-2/ja_wiki | -                     |
| English  | raw_data/en/en_xml.bz2 | processed/en/AA, processed/en/AB         | merged/en/en_merged.json                           | binarized/gpt-2/en_wiki | -                     |
