# python -m abci.dataset.wikipedia.wikidump_download

# Copyright 2021 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from IPython import embed
from urllib.request import urlretrieve

class Config(object):
    def __init__(self, lang="ja"):
        self.corpus_name = "{}_wiki".format(lang)

        # download
        # download_id = 20230501
        # self.download_link = "https://dumps.wikimedia.org/other/cirrussearch/{}/jawiki-{}-cirrussearch-content.json.gz".format(download_id, download_id)

        self.download_link = "https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2".format(lang, lang)
        self.raw_data_dir = "/bb/grandchallenge/gaf51090/datasets/wikipedia/raw_data/{}".format(lang)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        #self.raw_data_path = "{}/{}_wiki.json.gz".format(self.raw_data_dir, lang)
        self.raw_data_path = "{}/{}_xml.bz2".format(self.raw_data_dir, lang)

        # splitting into smaller files and convert them into loose json
        self.processed_data_dir = "/bb/grandchallenge/gaf51090/datasets/wikipedia/processed/{}".format(lang)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        self.chunk_size = "100M"

def download_data(config):
    if not os.path.exists(config.raw_data_path):
        print(f'Downloading {config.download_link} to {config.raw_data_path}')
        urlretrieve(config.download_link, config.raw_data_path)
        print(f'Successfully downloaded {config.raw_data_path}')

def preproces_data(config):
    input_file = config.raw_data_path
    output_dir = config.processed_data_dir
    chunk_size = config.chunk_size
    cmd = "python -m wikiextractor.WikiExtractor {} --o {} --bytes {} --json".format(input_file, output_dir, chunk_size)
    os.system(cmd)
    
if __name__ == "__main__":
    for lang in ("ja", "en"):
        config = Config(lang=lang)
        download_data(config)
        preproces_data(config)


