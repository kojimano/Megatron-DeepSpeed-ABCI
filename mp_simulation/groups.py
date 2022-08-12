import torch.distributed as dist

class MPU():
    def __init__(self, tp_world_size, ep_world_size=1, optim_a2a=False):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.tp_world_size = tp_world_size
        self.ep_world_size = ep_world_size
        if optim_a2a:
            assert self.ep_world_size % self.tp_world_size == 0

        for i in range(0, self.world_size, tp_world_size):
            ranks = range(i, i + tp_world_size)
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.tp_group = group

        if optim_a2a:
            for i in range(0, self.world_size, self.ep_world_size):
                for j in range(i, i+self.tp_world_size):
                    ranks = range(j, i+self.ep_world_size, self.tp_world_size)
                    group = dist.new_group(ranks)
                    if self.rank in ranks:
                        self.ep_group = group
        else:
            for i in range(0, self.world_size, ep_world_size):
                ranks = range(i, i + ep_world_size)
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self.ep_group = group

        for i in range(0, ep_world_size):
            ranks = range(i, self.world_size, ep_world_size)
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.ep_dp_group = group

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
