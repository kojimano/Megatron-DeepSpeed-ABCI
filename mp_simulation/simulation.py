import torch
import sys
import numpy as np
import argparse
import torch.distributed as dist


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

def allreduce(x, op=dist.ReduceOp.SUM):
    dist.all_reduce(x, op=op)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modify-rpl", action='store_true')
    parser.add_argument("--modify-cpl", action='store_true')
    parser.add_argument("--hsize", type=int)
    parser.add_argument("--bs-per-gpu", type=int, default=2)
    parser.add_argument("--reduce-scatter", action='store_true')

    args = parser.parse_args()
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    bs_per_gpu = args.bs_per_gpu
    modify_rpl=args.modify_rpl
    modify_cpl=args.modify_cpl

    hsize=args.hsize
    tp=dist.get_world_size()


    sq_len=2048
    flops = 16 * bs_per_gpu * sq_len * hsize ** 2 
    num_batches = 40


    mbs = bs_per_gpu * tp
    l1 = CPL(hsize, tp, modify_cpl).cuda().half()
    l2 = RPL(hsize, tp, modify_rpl, args.reduce_scatter).cuda().half()
    
    cpl_start = torch.cuda.Event(enable_timing=True)
    cpl_stop = torch.cuda.Event(enable_timing=True)

    cpl_bias_start = torch.cuda.Event(enable_timing=True)
    cpl_bias_stop = torch.cuda.Event(enable_timing=True)

    rpl_start = torch.cuda.Event(enable_timing=True)
    rpl_stop = torch.cuda.Event(enable_timing=True)


    rpl_bias_start = torch.cuda.Event(enable_timing=True)
    rpl_bias_stop = torch.cuda.Event(enable_timing=True)
    
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_stop = torch.cuda.Event(enable_timing=True)

    all_start = torch.cuda.Event(enable_timing=True)
    all_stop = torch.cuda.Event(enable_timing=True)

    cpl_flops = []
    rpl_flops = []
    total_flops = []

    batch = torch.randn(sq_len, mbs, hsize).cuda().half()
    for iter_no in range(num_batches):
        with torch.no_grad():
            all_start.record()
            cpl_start.record()
            out, bias = l1(batch)
            cpl_bias_start.record()
            out = GeLUFunction.apply(out, bias)
            cpl_bias_stop.record()
            cpl_stop.record()
            rpl_start.record()
            out, bias = l2(out)
            comm_start.record()
            if args.reduce_scatter:
                prev_out = out
                out = reduce_scatter(out)
            else:
                out = allreduce(out)
                msg_size = out.numel() * 2
            comm_stop.record()
            rpl_bias_start.record()
            if args.reduce_scatter:
                out = torch.add(out, bias, alpha=1/tp)
            else:
                out = out + bias
            rpl_bias_stop.record()
            if args.reduce_scatter:
                out = allgather(prev_out)
            rpl_stop.record()


            all_stop.record()

        torch.cuda.synchronize()
        rpl_time = rpl_start.elapsed_time(rpl_stop)
        rpl_tflops = (flops/2) * 1000 / 1e12 / rpl_time

        comm_time = comm_start.elapsed_time(comm_stop)

        rpl_bias_time = rpl_bias_start.elapsed_time(rpl_bias_stop)
        cpl_bias_time = cpl_bias_start.elapsed_time(cpl_bias_stop)

        cpl_time = cpl_start.elapsed_time(cpl_stop)
        cpl_tflops = (flops/2) * 1000 / 1e12 / cpl_time

        cpl_flops.append(cpl_tflops)
        rpl_flops.append(rpl_tflops)
        total_time = all_start.elapsed_time(all_stop)
        total_tflops = flops * 1000 / 1e12 / total_time

        if dist.get_rank() == 0:
            print(f"Iter No: {iter_no+1} : hsize = {hsize} tp = {tp} | CPL TFLOPs = {cpl_tflops:.2f} | RPL (modify_rpl={modify_rpl}) TFLOPs = {rpl_tflops:.2f} | Total TFLOPs = {total_tflops:.2f}")
            print(f"CPL time = {cpl_time:.2f} ms (matmul: {cpl_time-cpl_bias_time:.2f} ms bias: {cpl_bias_time:.2f}) | RPL time = {rpl_time:.2f} ms (matmul: {rpl_time-rpl_bias_time-comm_time:.2f} ms bias: {rpl_bias_time:.2f} ms comm: {comm_time:.2f} ms)  | Total time = {total_time:.2f} ms")
            msg_size_gb = msg_size / 1024 / 1024 / 1024
            time_in_sec = comm_time / 1000
            busbw = (msg_size_gb / time_in_sec) * (2 * (tp - 1) / tp) * 8 
            print(f"MSG Size = {msg_size_gb} GB | bw = {busbw} GbPS")
#    print(f"tp = {tp} | total Tflops = {flops /1e12} | CPL TFLOPs = {np.mean(cpl_flops[5:])} | RPL FLOPs = {np.mean(rpl_flops[5:])}")
