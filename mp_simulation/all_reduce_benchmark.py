import torch
import sys
import numpy as np
import argparse
import torch.distributed as dist
from groups import MPU
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsize", type=int)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--num-attempts", type=int, default=5)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    mpu = MPU(args.tp_size, 1)
    tp_size = mpu.get_model_parallel_world_size()
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    size = (12*args.num_layers*(args.hsize**2)//args.tp_size,)
    gradient_tensor = torch.rand(size, device='cuda', dtype=torch.float16)
    size_in_gbits = gradient_tensor.numel() * 2 * 8 / 1024 / 1024 / 1024
    dp_size = mpu.get_data_parallel_world_size()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for attempt_no in range(args.num_attempts):
        start_event.record()
        dist.all_reduce(gradient_tensor, group=mpu.get_data_parallel_group())
        end_event.record()
        torch.cuda.synchronize()
        time = start_event.elapsed_time(end_event) / 1000

        bus_bw_GbPS = 2 * (dp_size-1) / (dp_size) * size_in_gbits / time
        if mpu.get_world_rank() == 0:
            print(f"Attempt {attempt_no + 1} : bus_bw = {bus_bw_GbPS} Giga Bits Per Second, msg_size = {size_in_gbits} GigaBits, time = {time} s")
    
