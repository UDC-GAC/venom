/*
/*
 * Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "./blockwise_op.h"
#include "./blockwise_format.h"
#include "../common/library_util.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/Utils.h>

#include "ATen/cuda/CUDAContext.h"

#include "cuda_fp16.h"
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <chrono>

#include <cublas_v2.h>

using namespace at;
namespace spatha{

#define INSTANTIATE_FUNC(type, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
type##InitFn_t NAME_FUNC(type, Init, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##InitFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>; \
type##ExecFn_t NAME_FUNC(type, Exec, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##ExecFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>;

// block_sz = 32
INSTANTIATE_FUNC(SpmmNM, 32, 16, 16, 32, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 128, 32, 64, 128, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 128, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 64, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 32, 32, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 64, 32, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 64, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 128, 32, 64, 128, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 128, 32, 64, 128, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 128, 32, 128, 128, 16, 8, 32, 2);

// block_sz = 64
INSTANTIATE_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 4);


INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 64, 32, 64, 64, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 32, 128, 64, 32, 128, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 64, 32, 16, 32, 32, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 64, 16, 16, 64, 16, 16, 16, 8, 16, 2);

// block_sz = 128
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 4);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 5);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 6);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 4);


INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 64, 32, 16, 8, 32, 4);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 16, 128, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 16, 128, 32, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 16, 16, 64, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 16, 16, 128, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 128, 32, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 64, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 32, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 32, 32, 64, 16, 8, 32, 3);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 64, 64, 16, 8, 32, 3);

// block_sz = 256
INSTANTIATE_FUNC(SpmmNM, 256, 64, 16, 64, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 16, 64, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 16, 64, 64, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 4);

///
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 4);

//
INSTANTIATE_FUNC(SpmmNM, 128,32, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128,32, 32, 32, 32, 32, 16, 8, 32, 3);

INSTANTIATE_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128,128,32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128,128,32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 4);


//cublasHandle_t *handle=NULL;

torch::Tensor spmm_cuda(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias,
                        int A_num_rows,
                        int A_num_cols,
                        int B_num_cols,
                        int vec_length,
                        int n,
                        int m,
                        int nnz,
                        int seed,
                        int mbrow,
                        int brow)
{
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({B_num_cols,A_num_rows}, options);

    /* SpmmBlockwiseOp<ShapeBase<128, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,       // warp tile
                ShapeBase<16, 8, 32>,        // mma shape
                2>                           // number of pipeline stage
                op; */

    SpmmBlockwiseOp<ShapeBase<64, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,       // warp tile
                ShapeBase<16, 8, 32>,        // mma shape
                2>                           // number of pipeline stage
                op;

    op.initialize(A_num_rows, A_num_cols,
                  reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()),
                  reinterpret_cast<uint *>(A_metadata.data_ptr<int>()), reinterpret_cast<uint *>(A_indices.data_ptr<int>()),
                  B_num_cols,
                  reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
                  1.0f,
                  reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                  0.0f, m, n, brow, mbrow,
                  reinterpret_cast<half *>(bias.data_ptr<torch::Half>()));

    op();
    //cudaDeviceSynchronize();
    //printf("finished\n");
    /* if(handle==nullptr){
        handle = (cublasHandle_t*) malloc (sizeof(*handle));
        cublasCreate(handle);
        cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);
    } */

    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    //cublasMath_t original_math_mode;

    //TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &original_math_mode));
    //TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    /* float alpha = 1.0f;
    float beta = 0.0f;
    cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            B_num_cols, A_num_rows, A_num_cols/4,
            &alpha,
            reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()), CUDA_R_16F, A_num_cols,
            reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()), CUDA_R_16F, A_num_cols,
            &beta,
            reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()), CUDA_R_16F, B_num_cols,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); */

    //cublasDestroy(handle);

        /* cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        B_num_cols, A_num_rows, A_num_cols,
        &alpha,
        reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()), CUDA_R_16F, B_num_cols,
        reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()), CUDA_R_16F, A_num_cols,
        &beta,
        reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()), CUDA_R_16F, B_num_cols,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ); */

    return output_matrix;
}

}
