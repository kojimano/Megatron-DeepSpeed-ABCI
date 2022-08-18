import torch.distributed as dist
import torch
from collections import defaultdict

class MPU():
    def __init__(self, tp_world_size, ep_world_size=1, hier_a2a=False):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.tp_world_size = tp_world_size
        self.ep_world_size = ep_world_size

        for i in range(0, self.world_size, tp_world_size):
            ranks = range(i, i + tp_world_size)
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group

        for i in range(0, self.tp_world_size):
            ranks = range(i, self.world_size, self.tp_world_size)
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.dp_group = group
       
        ep_groups = []
        for i in range(0, self.world_size, ep_world_size):
            ranks = range(i, i + ep_world_size)
            ep_groups.append(ranks)
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.ep_group = group

        self.hier_a2a = hier_a2a
        if hier_a2a:
            num_gpus_per_node = 8
            for ep_group in ep_groups:
                ep_intra_node_groups = defaultdict(list)
                ep_inter_node_groups = defaultdict(list)
                for rank in ep_group:
                    ep_intra_node_groups[rank//num_gpus_per_node].append(rank)
                    ep_inter_node_groups[rank%num_gpus_per_node].append(rank)
                for intra_node_group in ep_intra_node_groups.values():
                    group=dist.new_group(intra_node_group)
                    if self.rank in intra_node_group:
                        self.ep_intra_node_group = group
                for inter_node_group in ep_inter_node_groups.values():
                    group=dist.new_group(inter_node_group)
                    if self.rank in inter_node_group:
                        self.ep_inter_node_group = group


        for i in range(0, ep_world_size):
            ranks = range(i, self.world_size, ep_world_size)
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.ep_dp_group = group

    def get_world_rank(self):
        return self.rank

    def get_model_parallel_rank(self):
        return self.rank % self.tp_world_size

    def get_model_parallel_world_size(self):
        return self.tp_world_size

    def get_data_parallel_rank(self):
        return self.rank // self.tp_world_size

    def get_data_parallel_world_size(self):
        return self.world_size // self.tp_world_size

    def get_data_parallel_group(self):
        return self.dp_group

    def get_model_parallel_group(self):
        return self.tp_group

    def get_expert_parallel_group(self):
        return self.ep_group

    def get_expert_world_size(self):
        return self.ep_world_size

    def all_to_all(self, in_, async_op=False):
        in_ = in_.contiguous()
        out_ = torch.empty_like(in_)
        assert not async_op
        if self.hier_a2a:
            assert not async_op
            dist.all_to_all_single(out_, in_, group=self.ep_intra_node_group)
            dist.all_to_all_single(out_, in_, group=self.ep_inter_node_group)
        else:
            dist.all_to_all_single(out_, in_, group=self.get_expert_parallel_group(), async_op=async_op)
        return out_
