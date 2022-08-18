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
    parser.add_argument("--mbs", type=int)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--ep-size", type=int)
    parser.add_argument("--hier", action="store_true")
    parser.add_argument("--async-op", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    #parser.add_argument("--drop-tokens", choices=["None", "local", "global", "global-opt"], default="None")

    args = parser.parse_args()
    

    dist.init_process_group(backend="nccl")
    mpu = MPU(args.tp_size, args.ep_size, hier_a2a=args.hier)
    tp_size = mpu.get_model_parallel_world_size()
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    mbs = args.mbs
    hsize=args.hsize
    tp=tp_size

    sq_len=2048
    torch.cuda.manual_seed(42)
    in_ = torch.randn(args.ep_size, sq_len*mbs//args.ep_size, hsize).cuda().half()
    #out1_ = torch.empty_like(in_)
    #out2_ = torch.empty_like(in_)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    net = torch.nn.Sequential(
            torch.nn.Linear(args.hsize, 4*args.hsize),
            torch.nn.ReLU(),
            torch.nn.Linear(4*args.hsize, args.hsize)).cuda().half()
    
    total_flops = 2 * (2 * mbs * sq_len * 4 * args.hsize * args.hsize) / 1e12
    
    if args.async_op or args.check_correctness:
        compute_stream = torch.cuda.default_stream()
        communication_stream = torch.cuda.Stream()
        a2a_events = [torch.cuda.Event(), torch.cuda.Event()]
        compute_events = [torch.cuda.Event(), torch.cuda.Event()]
    else:
        compute_stream = communication_stream = torch.cuda.default_stream()
    
    if args.check_correctness: 
        NUM_ATTEMPTS = 2
    else:
        NUM_ATTEMPTS = 100
   
    flops = []
    for _ in range(NUM_ATTEMPTS):
        if args.check_correctness:
            if _ < NUM_ATTEMPTS // 2:
                args.async_op = True
            else:
                args.async_op = False
                compute_stream = communication_stream = torch.cuda.default_stream()
                
        with torch.cuda.stream(communication_stream):
            start_event.record()

        for i in range(10):
            if args.async_op:
                ## split in_ out1_ and out2_ into two microbatches
                dim = 1
                split_size = in_.shape[dim] // 2
                ins_ = torch.split(in_, split_size, dim=dim)
                #out1s_ = torch.split(out1_, split_size, dim=dim)
                #out2s_ = torch.split(out2_, split_size, dim=dim)

        
                out_a2as = [] # will outputs of net

                #Step 1 - enqueue the incoming A2As
                with torch.cuda.stream(communication_stream):
                    for i in range(2):
                        out_a2as.append(mpu.all_to_all(ins_[i]))
                        a2a_events[i].record()

                #Step 2 - enqueue compute
                out = []
                with torch.cuda.stream(compute_stream):
                    for i in range(2):
                        compute_stream.wait_event(a2a_events[i])
                        out.append(net(out_a2as[i]))
                        compute_events[i].record()

                #Step 3 - enqueue the outgoing A2As
                final_outputs = []
                with torch.cuda.stream(communication_stream):
                    for i in range(2):
                        communication_stream.wait_event(compute_events[i])
                        final_outputs.append(mpu.all_to_all(out[i]))
                    
                    final_output = torch.cat(final_outputs, dim=dim)
                    
                    if args.check_correctness:
                        async_output = final_output
            else:
                out1_ = mpu.all_to_all(in_, async_op=False)
                out_ = net(out1_)
                out2_ = mpu.all_to_all(out_, async_op=False)
                if args.check_correctness:
                    sync_output = out2_
        #dist.all_to_all_single(out_, in_, group=mpu.get_expert_parallel_group())
        with torch.cuda.stream(communication_stream):
            end_event.record()
        torch.cuda.synchronize()

        this_flops = total_flops*1000/(start_event.elapsed_time(end_event)/10)

        if dist.get_rank() == 0:
            print(f"Attempt {_+1} : Async_OP = {args.async_op} : Avg FLOPs = {this_flops}") 
        flops.append(this_flops)

    if dist.get_rank() == 0:
        flops = flops[5:]
        if flops:
            print(f"Max FLOPs = {np.max(flops)}, Min FLOPs = {np.min(flops)}, Avg FLOPs = {np.mean(flops)}")
    if args.check_correctness:
        #assert torch.allclose(sync_output-async_output), f"Async {async_output.view(-1)[:5]} and Sync {sync_output.view(-1)[:5]} outputs did not match"
        if dist.get_rank() == 0:
            print(f"residual = {torch.abs(sync_output - async_output).mean()}")
            #print("The async implementation is correct")


