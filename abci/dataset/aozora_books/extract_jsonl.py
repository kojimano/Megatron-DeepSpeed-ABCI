# reference: https://qiita.com/Yupine/items/92d75865a72c60ae7285
# this script extract the text from aozora-books and format them into jsonl format simialr to the wikipedia data.
import re
import os, time, json
from tqdm import tqdm
from IPython import embed
from glob import glob
from bs4 import BeautifulSoup

class Config(object):
    def __init__(self):
        self.verbose = False
        self.raw_data_path = "/bb/grandchallenge/gaf51090/datasets/aozora_books/raw_data/aozorabunko/cards"
        self.output_path = "/bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books.jsonl"
        # self.output_path = "/bb/grandchallenge/gaf51090/datasets/aozora_books/processed/aozora_books_first200.jsonl"
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

def clean_text(text):
    # Remove ruby brackets, annotations, vertical bars, line breaks and certain characters
    text = re.sub(r'《.+?》|［＃.+?］|｜|\r\n|<!R>　', '', text)
    return text.strip()


def get_title_author(soup):
    # getting the title and the author of books
    title = soup.find("h1", class_="title") or soup.find("h1") or soup.find("H1")
    author = soup.find("h2", class_="author") or soup.find("h2") or soup.find("H2")
    if not title or not author:
        return None, None, False

    if title is None or author is None:
        return None, None, False
    else:
        return title.get_text(), author.get_text(), True

def get_main_text(soup):
    # getting the main text of books
    main_text = soup.find("div", class_='main_text') 
    
    # parsing books with irregular formatting
    irregular = False
    if not main_text:
        irregular = True
        main_text = soup.find("body")
        for yomigana in main_text.find_all(["h1","h2","H1","H2"]):
            yomigana.decompose()

    # merge and clean texts
    if not main_text:
        return None, False

    for yomigana in main_text.find_all(["rp","h4","rt"]):
        yomigana.decompose()
    sentences = [line.strip() for line in main_text.text.strip().splitlines()]
    merged_sentences = "\n".join(sentences)
    if irregular:
        if "底本：" not in merged_sentences:
            return None, False
        merged_sentences = merged_sentences.split("底本：")[0]

    return merged_sentences.strip(), True

def scrap(path_to_html, verbose=False):
    with open(path_to_html, 'rb') as html:
        soup = BeautifulSoup(html, 'lxml')

    # extracting the title and the author
    title, author, is_parsed = get_title_author(soup)
    if not is_parsed:
        return {}, False

    # summarizing meta data
    url = path_to_html.replace("/bb/grandchallenge/gaf51090/datasets/aozora_books/raw_data/aozorabunko", "https://github.com/aozorabunko/aozorabunko/blob/master")
    book = {
        "url": url,
        "title": title,
        "author": author
    }

    # processing the main text
    main_text, is_parsed = get_main_text(soup)
    if not is_parsed:
        return {}, False

    # attaching the title and author name to the main text.
    title, author, main_text = clean_text(title), clean_text(author), clean_text(main_text)
    book["text"] = title +  "\n" + author +  "\n\n" + main_text

    if verbose:
        print(book["text"])
        print(url)
        time.sleep(10)
        os.system("clear")

    return book, True

def format_to_jsonl(config, max_books=1e9):
    outfile  = open(config.output_path, 'w', encoding='utf-8') 
    def exit_function():
        outfile.close()
        print("{} books processed in {} seconds.".format(num_books, time.time() - st))
        print("Following html could not be parsed.")
        for html in error_htmls:
            print("\t{}".format(html))

    all_dirs = glob("{}/*/files".format(config.raw_data_path), recursive = True)

    # iterating all books
    st = time.time()
    num_books = 0
    error_htmls = []
    for dir in tqdm(all_dirs):
        for path in os.listdir(dir):
            if path.endswith(".html"):
                path = os.path.join(dir, path)
                entry, is_parsed = scrap(path, config.verbose)
                if is_parsed:
                    json.dump(entry, outfile)
                    outfile.write('\n')
                    num_books += 1
                else:
                    error_htmls.append(path)
                if num_books > max_books:
                    exit_function()
                    return
    exit_function()


if __name__ == "__main__":
    config = Config()
    format_to_jsonl(config)
