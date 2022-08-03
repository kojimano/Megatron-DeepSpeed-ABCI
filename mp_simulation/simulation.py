import torch
import sys
import numpy as np
import argparse
import torch.distributed as dist
from groups import MPU

@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g

def allreduce(x, mpu, op=dist.ReduceOp.SUM):
    dist.all_reduce(x, op=op, group=mpu.get_model_parallel_group())
    return x

def reduce_scatter(x):
    x = allreduce(x)
    tp_rank = dist.get_rank()
    tp_size = dist.get_world_size()
    size = x.shape[0] // tp_size
    input_list = list(torch.split(x, size, dim=0))
    output = input_list[tp_rank]
    return output

def allgather(x):
    x = allreduce(x) 
    return x


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp

class CPL(torch.nn.Module):
    def __init__(self, hsize, tp_size, modify=False):
        super().__init__()
        #assert 4*hsize % tp_size == 0
        if not modify:
            self.fc = torch.nn.Linear(hsize, 4*hsize//tp_size, bias=False)
        else:
            self.weight = torch.nn.Parameter(torch.randn(hsize, 4*hsize//tp_size))
        self.bias = torch.nn.Parameter(torch.zeros(4*hsize//tp_size))
        self.modify = modify

    def forward(self, x):
        if not self.modify:
            return self.fc(x), self.bias
        else:
            x_ = x.view(-1, x.shape[-1])
            y = torch.matmul(x_, self.weight)
            return y.view(x.shape[0], x.shape[1], -1), self.bias

class RPL(torch.nn.Module):
    def __init__(self, hsize, tp_size, modify=False, reduce_scatter=False):
        super().__init__()
#        assert 4*hsize % tp_size == 0
        if not modify:
            self.fc = torch.nn.Linear(4*hsize//tp_size, hsize, bias=False)
        else:
            self.weight = torch.nn.Parameter(torch.randn(4*hsize//tp_size, hsize))
        #if not reduce_scatter:
        self.bias = torch.nn.Parameter(torch.zeros(hsize))
        #else:
        #    self.bias = torch.nn.Parameter(torch.zeros(hsize//tp_size))
        self.modify = modify
        self.reduce_scatter = reduce_scatter
        self.tp_size = tp_size

    def forward(self, x):
        if not self.modify:
            out = self.fc(x)
        else:
            x_ = x.view(-1, x.shape[-1])
            y = torch.matmul(x_, self.weight)
            out  = y.view(x.shape[0], x.shape[1], -1)
        if self.reduce_scatter:
            return out, self.bias 
        else:
            return out, self.bias

class ParallelMLP(torch.nn.Module):
    def __init__(self, hsize, tp_size, mpu, modify_cpl=False, modify_rpl=False, reduce_scatter=False):
        super().__init__()

        self.cpl = CPL(hsize, tp_size, modify_cpl)
        self.rpl = RPL(hsize, tp_size, modify_rpl, reduce_scatter)

        self.cpl_start = torch.cuda.Event(enable_timing=True)
        self.cpl_stop = torch.cuda.Event(enable_timing=True)

        self.cpl_bias_start = torch.cuda.Event(enable_timing=True)
        self.cpl_bias_stop = torch.cuda.Event(enable_timing=True)

        self.rpl_start = torch.cuda.Event(enable_timing=True)
        self.rpl_stop = torch.cuda.Event(enable_timing=True)

        self.rpl_bias_start = torch.cuda.Event(enable_timing=True)
        self.rpl_bias_stop = torch.cuda.Event(enable_timing=True)
        
        self.comm_start = torch.cuda.Event(enable_timing=True)
        self.comm_stop = torch.cuda.Event(enable_timing=True)

        self.reduce_scatter = reduce_scatter
        self.mpu = mpu

    def forward(self, x):
        self.cpl_start.record()
        
        out, bias = self.cpl(batch)
        
        self.cpl_bias_start.record()
        out = GeLUFunction.apply(out, bias)
        self.cpl_bias_stop.record()
        
        self.cpl_stop.record()
        

        self.rpl_start.record()
        out, bias = self.rpl(out)
        self.comm_start.record()
        if self.reduce_scatter:
            prev_out = out
            out = reduce_scatter(out, self.mpu)
        else:
            out = allreduce(out, self.mpu)
        self.comm_stop.record()

        self.rpl_bias_start.record()
        if args.reduce_scatter:
            out = torch.add(out, bias, alpha=1/tp)
        else:
            out = out + bias
        self.rpl_bias_stop.record()
        
        if args.reduce_scatter:
            out = allgather(prev_out, self.mpu)
        self.rpl_stop.record()
        
        return out

    def profile_stats(self):
        rpl_time = self.rpl_start.elapsed_time(self.rpl_stop)

        comm_time = self.comm_start.elapsed_time(self.comm_stop)

        rpl_bias_time = self.rpl_bias_start.elapsed_time(self.rpl_bias_stop)
        cpl_bias_time = self.cpl_bias_start.elapsed_time(self.cpl_bias_stop)

        cpl_time = self.cpl_start.elapsed_time(self.cpl_stop)

        total_time = self.cpl_start.elapsed_time(self.rpl_stop)

        print(f"CPL time = {cpl_time:.2f} ms (matmul: {cpl_time-cpl_bias_time:.2f} ms bias: {cpl_bias_time:.2f}) | RPL time = {rpl_time:.2f} ms (matmul: {rpl_time-rpl_bias_time-comm_time:.2f} ms bias: {rpl_bias_time:.2f} ms comm: {comm_time:.2f} ms)  | Total time = {total_time:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modify-rpl", action='store_true')
    parser.add_argument("--modify-cpl", action='store_true')
    parser.add_argument("--hsize", type=int)
    parser.add_argument("--bs-per-gpu", type=int, default=2)
    parser.add_argument("--reduce-scatter", action='store_true')
    parser.add_argument("--tp-size", type=int)


    args = parser.parse_args()

    assert not args.reduce_scatter, "not sure if this is correct"
    dist.init_process_group(backend="nccl")
    mpu = MPU(args.tp_size)
    tp_size = mpu.get_model_parallel_world_size()
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    bs_per_gpu = args.bs_per_gpu
    modify_rpl=args.modify_rpl
    modify_cpl=args.modify_cpl

    hsize=args.hsize
    tp=tp_size

    sq_len=2048
    flops = 16 * bs_per_gpu * sq_len * hsize ** 2
    num_batches = 40
    num_gpus = dist.get_world_size()

    mbs = bs_per_gpu * tp
    
    model = ParallelMLP(hsize, 
            tp_size, 
            mpu,
            modify_cpl=args.modify_cpl, 
            modify_rpl=args.modify_rpl, 
            reduce_scatter=args.reduce_scatter).cuda().half()

    batch = torch.randn(sq_len, mbs, hsize).cuda().half()
    batch_start = torch.cuda.Event(enable_timing=True)
    batch_end = torch.cuda.Event(enable_timing=True)
    for iter_no in range(num_batches):
        batch_start.record()
        with torch.no_grad():
            model(batch)
        batch_end.record()
        torch.cuda.synchronize()
        batch_time = batch_start.elapsed_time(batch_end) / 1000
        tflops_per_gpu = flops / 1e12 / batch_time
        if dist.get_rank() == 0:
            print(f"Iteration {iter_no+1} : TFLOPs: {tflops_per_gpu}")
            model.profile_stats()
