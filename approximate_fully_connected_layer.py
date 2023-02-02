import torch
import torch.nn as nn
import torch.nn.functional as F
import cuda_layers
import cpp_layers
import pdb

import math

class linear_appx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias):
        ctx.save_for_backward(*X, weight, bias)

        #(m, n) = X.shape
        (k, _) = weight.shape
        
        if (X[0].is_cuda == True):

            # Must flatten the weights so CUDA sees them sequentially.
            weight = torch.transpose(weight, 0, 1).flatten()
            out = cuda_layers.linear_forward(X, weight, bias, k)#m, n, k)

            return out.reshape(-1, max([x.shape[1] for x in X]), max([x.shape[2] for x in X]), k)
        else:
            return cpp_layers.linear_forward(X, weight, bias)

            # Manually calculate FC for debug purposes, if needed.
            # output = torch.empty(m, k, device='cpu')
            # for k in range(m):
            #     for l in range(k):
            #         accumulation = 0.0
            #         for j in range(n):
            #             accumulation += X[k, j] * weight[j, l]
            #         output[k, l] = accumulation + bias[l]
            #
            # return output

    @staticmethod
    def backward(ctx, grad_output):
        (*inputs), weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = []
            for input in inputs:
                #grad_output_down = F.interpolate(grad_output.permute(0, 3, 1, 2), size = input.shape[1:3], mode="nearest").permute(0, 2, 3, 1)
                grad_output_down = F.max_pool2d(grad_output.permute(0, 3, 1, 2), kernel_size=grad_output.shape[1]//input.shape[1], stride=grad_output.shape[1]//input.shape[1], padding=0).permute(0, 2, 3, 1)
                grad_weight += [grad_output_down.reshape(-1, grad_output_down.shape[-1]).t().mm(input.reshape(-1, input.shape[-1]))]
            grad_weight = torch.cat(grad_weight, dim=-1)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features ):
        super(MyLinear, self).__init__()
        self.fn = linear_appx.apply
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.init_weights()
    
    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.fn(x, self.weight, self.bias)
        return x
