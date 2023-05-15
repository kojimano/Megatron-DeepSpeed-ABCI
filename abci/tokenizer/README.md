# Train tokenizer
## Training Dataset
| Dataset Source| Size (GB) |
| - | - |
| CC (ja) | TBW |
| Wikipedia (ja) | TBW |
| Wikipedia (en) | TBW |
| RedPajama Github | TBW |
## Special Tokens
| |  |
| - | - |
| `<\|endoftext\|>` | Represents EOS, BOS. |
| `<\|endofline\|>` | Represents `\n`. For the LM training, you have to preprocess your training dataset replacing `\n` to `<\|endofline\|>`. |
| `<\|padding\|>` | Represents padding. Not used during the model training. |
## Training Steps

### Train Sentencepiece Models
For `nmt_nfkc` normalization model : 
```
$ ./spm_train --input=/home/kurita/abci-tokenization/dataset/spm_input_replaced_all.txt  --model_prefix=/home/kurita/abci-tokenization/dataset/spm_input_fall_replaced_all_wodummyprefix --vocab_size=50000 --character_coverage=1.0 --model_type=unigram --byte_fallback=true --train_extremely_large_corpus=true --unk_id=0 --bos_id=-1 --eos_id=-1 --pad_id=-1 --user_defined_symbols="<|endoftext|>","<|endofline|>","<|padding|>" --add_dummy_prefix=false
```
For `identity` normalization model :
```
$ ./spm_train --input=/home/kurita/abci-tokenization/dataset/spm_input_replaced_all.txt  --model_prefix=/home/kurita/abci-tokenization/dataset/spm_input_fall_replaced_all_identity_wodummyprefix --vocab_size=50000 --character_coverage=1.0 --model_type=unigram --byte_fallback=true --train_extremely_large_corpus=true --unk_id=0 --bos_id=-1 --eos_id=-1 --pad_id=-1 --user_defined_symbols="<|endoftext|>","<|endofline|>","<|padding|>" --normalization_rule_name=identity --add_dummy_prefix=false
```
**Note:the input dataset `spm_input_replaced_all` is only available on the internal server, not on ABCI**
### Modify Vocab & Normalization Setting Manually
TBW
## Available Tokenizers
- The `nmt_nfkc` normalization model
    - model : `/bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_wodummyprefix_modified.model`
    - vocab : `/bb/grandchallenge/gaf51090/tokenizer_new/nmt_nfkc_vocab.txt`
- The `identity` model
    - model : `/bb/grandchallenge/gaf51090/tokenizer_new/spm_input_fall_replaced_all_identity_wodummyprefix_modified.model`
    - vocab : `/bb/grandchallenge/gaf51090/tokenizer_new/idendity_vocab.txt`

## Test tokenization
```
source .your_venv/bin/activate # activate python venv
pip install sentencepiece
>>> import sentencepiece as spm
>>> sp = spm.SentencePieceProcessor(model_file='test.model')
>>> sp.decode(sp.encode('吾輩は猫である'))
吾輩は猫である
```

## Check vocab
```
$ less vocab_file_path
```