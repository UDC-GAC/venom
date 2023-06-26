#
# Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import statistics
import subprocess
import ctypes

import sten

import math
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from pathlib import Path

import timeit
import argparse
import sys
import time

from grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify

import spatha

parser = argparse.ArgumentParser()

parser.add_argument('-m', type=int, default=16)
parser.add_argument('-n', type=int, default=2)
parser.add_argument('-v', type=int, default=128)

parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--sparsetime', action='store_true', default=False)

args = parser.parse_args()

m          = args.m
n          = args.n
v          = args.v
sparsetime = args.sparsetime


class NMVectorSparsifier:
    def __init__(self, n, m, tileM):
        self.n = n
        self.m = m
        self.tileM = tileM

    def __call__(self, tensor, grad_fmt=None):
        # uncomment to use magnitude-pruning -> mask, columns
        #mask, columns = nm_vector_mask_sparsify(tensor, sparsifier.n, sparsifier.m, sparsifier.tileM)

        # uncomment to use random pruning (cuSparseLt-like approach) -> mask, columns
        nrows, ncols = tensor.shape
        columns = torch.zeros(nrows//self.tileM, ncols//self.m*4, dtype=torch.int32)
        columns = columns.reshape((-1,4)) + torch.tensor([0,1,2,3], dtype=torch.int32)
        columns = columns.reshape((nrows//self.tileM, ncols//self.m*4))

        mask = torch.zeros(tensor.shape, dtype=tensor.dtype)
        m = torch.cat( (torch.tensor([1,0,1,0]), torch.zeros(self.m-4)), 0 )
        mask = mask.reshape(-1, self.tileM, self.m) + m
        mask = mask.reshape(tensor.shape)

        sparse_mtx = sten.SparseTensorWrapper.wrapped_from_dense(
            SrNMTensor(self.n, self.m, self.tileM, tensor, mask, columns, tensor.device),
            tensor,
            grad_fmt,
        )

        return sparse_mtx

def sparse_dense_mul_dispatch(sparse_values, sparse_indices, sparse_metadata, dense, nrows_sp, ncols_sp, ncols_d, m, n, v, nnz, bias):

    dense_ = dense.contiguous()

    output = spatha.spmm(sparse_metadata,  # metadata
                          sparse_indices,   # indices
                          sparse_values,    # values
                          dense_,           # rhs_matrix
                          bias,
                          nrows_sp,         # A_num_rows
                          ncols_sp,         # A_num_cols
                          ncols_d,          # B_num_cols
                          v,                # vec_length
                          n,                # n
                          m,                # m
                          nnz,              # nnz
                          0,                # seed
                          32,               # mbrow
                          4                 # brow
                          )

    return output


class SrnmSpmm(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear):
        super().__init__()
        self.bias = original.bias

        # Convert weights from original module to SrNM
        w = NMVectorSparsifier(n, m, v)(original.weight).wrapped_tensor

        self.values = torch.nn.Parameter(w.values)
        #self.columns = self.register_buffer('columns', w.columns)
        self.columns = w.columns
        self.metadata = w.metadata

        self.nrows_sp = w.nrows
        self.ncols_sp = w.ncols
        self.nnz      = w.nnz

    def forward(self, input):

        flattened_input = torch.flatten(input, start_dim=0, end_dim=-2)

        ncols_d  = flattened_input.T.shape[1]
        DM, _    = flattened_input.shape

        output = sparse_dense_mul_dispatch(self.values, self.columns, self.metadata, flattened_input.T, self.nrows_sp, self.ncols_sp,
                                           ncols_d, m, n, v, self.nnz, self.bias)
        output = output.reshape((*input.shape[0:-1], -1))[..., :DM]
        return output

def report_time(name, data, number):
    for d in data:
        time_ms = d / number * 1000
        #print(f'n {n} m {m} format {name} time_ms {time_ms:.3f}')
    ds = [(d / number * 1000) for d in data]
    mean = statistics.mean(ds)
    median = statistics.median(ds)
    std = statistics.stdev(ds)

    if name == "n:m":
        cfg = str(n)+","+str(m)+","
    else:
        cfg = "0,0,"
    print(
        "0,"+cfg+str(v)+","+str(mean)+","+str(median)+","+str(std)+","+str(len(ds))
    )

def linear_to_spmm(mod, weights_to_sparsify):
    if isinstance(mod, torch.nn.Linear):
        return SrnmSpmm(mod)

    for name, m in mod.named_children():
        if isinstance(m, SrnmSpmm):
            continue
        if isinstance(m, torch.nn.Linear):
            setattr(mod, name, SrnmSpmm(m))
        elif m is not mod:
            linear_to_spmm(m, weights_to_sparsify)

    return mod

def transformer_encoder_layer_prototype(num_repeats, number):
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-uncased')
    model2 = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-large-uncased').to(device='cuda:0').half()

    input = torch.randint(low=0, high=100, size=(32, 512))#, dtype=torch.half)

    weights_to_sparsify = [
        module
        for module_name, module in model.named_modules()
        if (
            isinstance(module, torch.nn.modules.linear.Linear)
            and "encoder.layer" in module_name
        )
    ]
    model = model.to(device='cuda:0').half()
    input = input.to(device='cuda:0')

    sparse_model = linear_to_spmm(model, weights_to_sparsify)

    # execute once to fill caches
    #print("Starting execution")
    output = sparse_model(input)

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            with record_function("model_inference"):
                output = sparse_model(input)
        prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
        prof.export_chrome_trace("trace_sparse.json")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        #exit()

        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            with record_function("model_inference"):
                output = model2(input)
        prof.export_stacks("/tmp/profiler_stacks_dense.txt", "self_cuda_time_total")
        prof.export_chrome_trace("trace_dense.json")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        exit()

    #warmup
    timeit.repeat('output = model2(input)', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('output = model2(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('dense', dense_times, number)

    #warmup
    timeit.repeat('output = sparse_model(input)', repeat=10, number=number, globals=locals())
    sparse_times = timeit.repeat('output = sparse_model(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('n:m', sparse_times, number)

if __name__ == "__main__":
    torch. set_grad_enabled(False)
    transformer_encoder_layer_prototype(num_repeats=30, number=1)