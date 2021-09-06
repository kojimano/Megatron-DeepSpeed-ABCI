import os
from typing import Dict
import torch 

ZERO_FILE_PREFIX = 'zero_pp_rank_'
LAYER_FILE_PREFIX = 'layer_'
MP_RANK_FILE_PREFIX = 'mp_rank_'
EMBEDDING_LAYER_INDEX = 0
FINAL_LAYER_NORM_INDEX = -1
class DeepSpeedCheckpoint(object):
    def __init__(self, dir):
        self.dir = dir
        self.file_list = self._get_files(dir)
        self.zero_files = self._get_files_with_prefix(self.file_list, ZERO_FILE_PREFIX)
        self.layer_files = self._get_files_with_prefix(self.file_list, LAYER_FILE_PREFIX)
        self.mp_rank_files = self._get_files_with_prefix(self.file_list, MP_RANK_FILE_PREFIX)
        self.layer_keys = self._get_layer_keys()
        self.layer_count = len(self.layer_keys)
        self.tp_degree = len(self._get_files_with_prefix(self.layer_files, f'{LAYER_FILE_PREFIX}01'))
        self.pp_degree = len(self.mp_rank_files) // self.tp_degree
        self.dp_degree = len(self.zero_files) // (self.pp_degree * self.tp_degree)
        self._sanity_check()
        self.pp_to_transformer_map = self._build_pp_transformer_map()
        self.transformer_file_map = self._build_transformer_file_map()
        self.tp_to_embedding_map = self._build_tp_other_layer_map(EMBEDDING_LAYER_INDEX)
        self.tp_to_final_norm_map = self._build_tp_other_layer_map(FINAL_LAYER_NORM_INDEX)


    def show_layer_file_map(self):
        self._dump_file_map(self.layer_file_map)

    def show_tp_embedding_map(self):
        self._dump_mapping(self.tp_to_embedding_map, 'tp_to_embedding_layers')

    def show_tp_final_norm_map(self):
        self._dump_mapping(self.tp_to_final_norm_map, 'tp_to_final_norm_layers')

    def show_pp_tranformer_map(self):
        self._dump_mapping(self.pp_to_transformer_map, 'pp_to_tranformer_layers')

    def show_transformer_file_map(self):
        self._dump_mapping(self.transformer_file_map, 'rank_to_tranformer_files')

    def get_embedding_state(self, tp_index: int) -> Dict:
        embedding_files = self._get_files_with_prefix(self.layer_files, self.layer_keys[EMBEDDING_LAYER_INDEX])
        assert tp_index < len(embedding_files)
        sd = torch.load(embedding_files[tp_index])
        return sd

    def get_transformer_state(self, tp_index: int, pp_index: int) -> list:
        assert tp_index < self.tp_degree
        assert pp_index < self.pp_degree
        t_list = []
        for fname in self.transformer_file_map[(tp_index, pp_index)]:
            sd = torch.load(fname)
            t_list.append(sd)
        return t_list   

    def get_final_norm_state(self, tp_index:int) -> Dict:
        final_norm_files = self._get_files_with_prefix(self.layer_files, self.layer_keys[FINAL_LAYER_NORM_INDEX])
        assert tp_index < len(final_norm_files)
        sd = torch.load(final_norm_files[tp_index])
        return sd

    def _build_tp_other_layer_map(self, layer_index:int):
        assert layer_index < len(self.layer_files)
        layer_files = self._get_files_with_prefix(self.layer_files, self.layer_keys[layer_index])
        data_map = {i:fname for i, fname in enumerate(layer_files)}
        return data_map

    def _build_pp_transformer_map(self):
        data_map = {}
        transformer_layers = self.layer_keys[1:-1]
        layers_per_pp = len(transformer_layers) // self.pp_degree
        data_map = {i:transformer_layers[i*layers_per_pp:(i+1)*layers_per_pp] for i in range(0, self.pp_degree)}
        return data_map

    def _dump_mapping(self, data_map, map_tag = None):
        if map_tag is not None:
            print(f'Dump mapping: {map_tag}')
        for k, v in data_map.items():
            print(f'{k} = {v}')

    def _build_transformer_file_map(self):
        transformer_layer_keys = self.layer_keys[1:-1]
        file_map = {}
        layers_per_pp = len(transformer_layer_keys) // self.pp_degree
        for key_index, layer_key in enumerate(transformer_layer_keys):
            pp_index = key_index // layers_per_pp
            layer_files = self._get_files_with_prefix(self.layer_files, layer_key)
            assert len(layer_files) == self.tp_degree
            for file_index, fname in enumerate(layer_files):
                map_key = (file_index, pp_index)
                if not map_key in file_map.keys():
                    file_map[map_key] = []
                file_map[map_key].append(fname)
        
        return file_map
        
    def _sanity_check(self):
        assert len(self.mp_rank_files) % self.tp_degree == 0
        assert len(self.zero_files) % (self.pp_degree * self.tp_degree) == 0
        assert len(self.layer_keys) > 2
        assert (len(self.layer_keys) - 2) % self.pp_degree == 0
     
    def _get_files_with_prefix(self, all_files, prefix):
        file_list = []
        for file_path in all_files:
            _, fname = os.path.split(file_path)
            if fname.startswith(prefix):
                file_list.append(file_path)
        
        return sorted(file_list)

    def validate_files(self):
        for file in self.file_list:
            if not os.path.isfile(file):
                print(f'Error: {file} is not existent')
        
    def _get_files(self, dir):
        file_list = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def _get_layer_keys(self):
        key_set = set()
        key_len = len(LAYER_FILE_PREFIX) + 2 
        for file_path in self.layer_files:
            _, fname = os.path.split(file_path)
            key_set.add(fname[:key_len])
        return sorted(list(key_set))
    