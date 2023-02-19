lr=3.0e-4
train_tokens_in_billion=300
bash ds_pretrain_gpt_350M_moe_base_script.sh ${lr} \
    ${train_tokens_in_billion}