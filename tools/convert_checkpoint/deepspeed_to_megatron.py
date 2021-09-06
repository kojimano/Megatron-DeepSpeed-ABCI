import argparse
import os
import torch
from collections import OrderedDict
from deepspeed_checkpoint import DeepSpeedCheckpoint

MODEL_KEY = 'model'
LANGUGAGE_MODEL_KEY = 'language_model'
EMBEDDING_KEY = 'embedding'
ENCODER_KEY = 'encoder'
WORD_EMBEDDINGS_FOR_HEAD_KEY = 'word_embeddings_for_head'
WORD_EMBEDDINGS_KEY = 'word_embeddings'
FINAL_LAYER_NORM_KEY ='final_layernorm'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input DeepSpeed Checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output Megatron checkpoint folder')
    args = parser.parse_args()
    print(f'args = {args}')
    return args 


def _convert_ds_transformer_state(sd_list):
    new_sd = OrderedDict()
    for i, sd in enumerate(sd_list):
        for key, value in sd.items():
            new_key = f'layers.{i}.{key}'
            new_sd[new_key] = value

    return new_sd 

def _create_checkpoint_path(tp_index, pp_index):
    rank_folder = f'mp_rank_{tp_index:02d}_{pp_index:03d}'
    return os.path.join(rank_folder, 'model_optim_rng.pt')

def _create_megatron_dict():
    language_model_dict = {
        EMBEDDING_KEY: {},
        ENCODER_KEY: {}
    }
    megatron_dict = {
        MODEL_KEY: {LANGUGAGE_MODEL_KEY: language_model_dict},
        WORD_EMBEDDINGS_FOR_HEAD_KEY: OrderedDict()
    }
    return megatron_dict


def _save_checkpoint(file_path, chkpt_sd):
    dir, _ = os.path.split(file_path)
    os.makedirs(dir, exist_ok=True)
    torch.save(chkpt_sd, file_path)

def _create_rank_checkpoint(ds_checkpoint, output_folder, tp_index, pp_index):
    checkpoint_path = os.path.join(output_folder, _create_checkpoint_path(tp_index, pp_index))

    meg_encoder_sd = OrderedDict() 
    meg_embedding_sd = OrderedDict()
    meg_embedding_for_head_sd = OrderedDict()

    transformer_sd = ds_checkpoint.get_transformer_state(tp_index, pp_index)
    meg_encoder_sd.update(_convert_ds_transformer_state(transformer_sd))

    if pp_index in [0, ds_checkpoint.pp_degree - 1]:
        embedding_sd = ds_checkpoint.get_embedding_state(tp_index)
        if pp_index == 0:
            meg_embedding_sd.update(embedding_sd)

        if pp_index == ds_checkpoint.pp_degree -1:
            for key, value in embedding_sd.items():
                if key.startswith(WORD_EMBEDDINGS_KEY):
                    fields = key.split('.')
                    new_fields = fields[1:]
                    new_key = '.'.join(new_fields)
                    meg_embedding_for_head_sd[new_key] = value
            
            final_norm_sd = ds_checkpoint.get_final_norm_state(tp_index)
            new_final_norm_sd = {f'{FINAL_LAYER_NORM_KEY}.{key}': value for key, value in final_norm_sd.items()}
            meg_encoder_sd.update(new_final_norm_sd)

    checkpoint_sd = _create_megatron_dict()
    checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][EMBEDDING_KEY] = meg_embedding_sd
    checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][ENCODER_KEY] = meg_encoder_sd
    checkpoint_sd[MODEL_KEY][WORD_EMBEDDINGS_FOR_HEAD_KEY] = meg_embedding_for_head_sd

    _save_checkpoint(checkpoint_path, checkpoint_sd)


def main():
    print(f'Convert DeepSpeed Checkpoint to Megatron Checkpoint')
    
    args = parse_arguments()
    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to Megatron checkpoint in {args.output_folder}')

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder)
    for i in range(0, ds_checkpoint.tp_degree):
        for j in range(0, ds_checkpoint.pp_degree):
            _create_rank_checkpoint(ds_checkpoint, args.output_folder, i, j)


if __name__ == "__main__":
    main()
