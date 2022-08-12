import torch
import sys
import numpy as np
import argparse
import torch.distributed as dist
from groups import MPU
import time

class CPL(torch.nn.Module):
    def __init__(self, hsize, tp_size, modify=False):
        super().__init__()
        #assert 4*hsize % tp_size == 0
        if not modify:
            self.fc = torch.nn.Linear(hsize, 3*hsize//tp_size, bias=False)
        else:
            self.weight = torch.nn.Parameter(torch.randn(hsize, 3*hsize//tp_size))
        self.bias = torch.nn.Parameter(torch.zeros(3*hsize//tp_size))
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
        #assert tp_size == 1
        if not modify:
            self.fc = torch.nn.Linear(hsize//tp_size, hsize, bias=False)
        else:
            self.weight = torch.nn.Parameter(torch.randn(hsize//tp_size, hsize))
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
    #parser.add_argument("--modify-rpl", action='store_true')
    #parser.add_argument("--modify-cpl", action='store_true')
    parser.add_argument("--hsize", type=int)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--add-bias", action='store_true', default=False)
    #parser.add_argument("--tp-size", type=int)
    #parser.add_argument("--ep-size", type=int)
    #parser.add_argument("--enable-expert-tensor-parallelism", action='store_true')
    parser.add_argument("--layer-type", choices=["None", "RPL", "CPL"])

    args = parser.parse_args()
    tp = 4
    mbs = args.mbs
    hsize = args.hsize
    sq_len = 2048
    num_iters = 10
    layer_type = "RPL"

    if layer_type == "RPL":
        layer = CPL(hsize=hsize, tp_size=tp).half().cuda()
    elif layer_type == "CPL":
        layer = CPL(hsize=hsize, tp_size=tp).half().cuda()
    else:
        raise NotImplementedError

    flops = 16 * mbs * sq_len * hsize ** 2 / tp / 2
    batch = torch.randn(sq_len * mbs, hsize).cuda().half()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(num_iters):
        start_event.record()
        with torch.no_grad():
            out, bias = layer(batch)
            if args.add_bias:
                out = out + bias
        end_event.record()
        torch.cuda.synchronize()
        time = start_event.elapsed_time(end_event) / 1000
        tflops = flops / 1e12 / time
        print(f"Iteration {i+1}: {layer_type} Layer | mbs = {mbs} | TFLOPs: {tflops}")


