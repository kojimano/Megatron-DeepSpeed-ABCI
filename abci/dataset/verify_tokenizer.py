# python -m abci.dataset.verify_tokenizer
import os
import json
import time
import random
import difflib
from IPython import embed
from megatron.tokenizer import _JapaneseSentencePiece


datasets = {
    #"ja_wiki": "/bb/grandchallenge/gaf51090/datasets/wikipedia/merged/ja/ja_merged.json",
    #"en_wiki": "/bb/grandchallenge/gaf51090/datasets/wikipedia/merged/en/en_merged.json",
    #"aozora": "/bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl",
    "code": "/bb/grandchallenge/gaf51090/datasets/redpajama_github/merged/merged.jsonl",
    #"blog": "/bb/grandchallenge/gaf51090/datasets/ameba_blog_small_for_stress_testing/entry_text.jsonl",
}

if __name__ == "__main__":
    vocab_file = "/bb/grandchallenge/gaf51090/datasets/tokenization_replaced/spm_input_fall_replaced_all.model"
    max_samples = 5
    tokenizer = _JapaneseSentencePiece(vocab_file=vocab_file)

    for ds_key in datasets.keys():
        ds_path = datasets[ds_key]
        jsonl = open(ds_path, 'r', encoding='utf-8')
        num_correct, num_incorrect = 0, 0

        while True:
            l = jsonl.readline()
            data = json.loads(l)
            text = data["text"]
            if text == '':
                continue
            if random.random() < 0.99:
                continue

            # tokenize
            ids = tokenizer.tokenize(text)
            reconstructed_text = tokenizer.detokenize(ids)
            if text != reconstructed_text:
                print("="*40 + "correct: {} incorrect: {}".format(num_correct, num_incorrect) + "="*40)
                original_lines = text.split("\n")
                reconstructed_lines = reconstructed_text.split("\n")
                if len(original_lines) == len(reconstructed_lines):
                    for i in range(len(original_lines)):
                        if original_lines[i] != reconstructed_lines[i]:
                            print(original_lines[i])
                            print(reconstructed_lines[i])
                            print("*"*40)
                num_incorrect += 1

                """
                print_side_by_side(text, reconstructed_text)
                print("="*40 + "correct: {} incorrect: {}".format(num_correct, num_incorrect) + "="*40)
                print(text)
                print("="*100)
                print(reconstructed_text)
                print("="*100)
                matcher = difflib.SequenceMatcher(a=text, b=reconstructed_text)
                for match in matcher.get_matching_blocks():
                    print("Match             : {}".format(match))
                    print("Matching Sequence : {}".format(text[match.a:match.a+match.size]))
                num_incorrect += 1
                """
                embed()
                os.system("clear")
                #time.sleep(10)
            else:
                num_correct += 1

            if num_incorrect >= max_samples:
                break





