checkpoint_path="/blob/users/conglli/project/data_efficient_gpt/checkpoint/gpt_0.35B_tok300B_lr3.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs256_mbs4_g64_seed1234_ep64/global_step572205/"
hostname_and_rank="worker-0:0"
master_port=12345
ep_size=1 # same as number of ranks in hostname_and_rank
batch_size=2
num_fewshot=0
tasks="lambada"

bash ds_evalharness_moe.sh ${checkpoint_path} ${hostname_and_rank} ${master_port} ${ep_size} ${batch_size} ${num_fewshot} ${tasks} &
