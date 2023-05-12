import json
import argparse
from pathlib import Path
from tqdm import tqdm

REPLACE_CHAR = "<|endofline|>"


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", help="Path to the input file. Each line should be a json object with a 'text' key.")
    parser.add_argument("output_path",help="Path to the output file. Each line will be a document with line breaks replaced by the replace_char.")
    parser.add_argument("--replace_char", default=REPLACE_CHAR, help="The character to replace line breaks with. Default: {}".format(REPLACE_CHAR)

    args = parser.parse_args()

    return args


def replace_breaks():
    args = get_args()
    with Path(args.input_path).open("r") as input_file:
        with Path(args.output_path).open("w") as output_file:
            for line in tqdm(input_file):
                line = json.loads(line)
                output_file.write(line["text"].replace("\n", args.replace_char) + "\n")


if __name__ == "__main__":
    replace_breaks()
