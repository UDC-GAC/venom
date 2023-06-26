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

#pragma once

#include "vector.h"
#include "memcpy.h"
#include "mma.h"
namespace spatha {

template<
    int block_M, int block_N, int blockDim, int warp_M,
    typename CSwizzle, bool GridMappingXYToMN
>
__device__ __forceinline__
void epilogue_impl
(const int M, const int N, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
#ifdef OUT_32B
    const int kAccess = 8;
    const int smem_ldc = block_N;
    int ldc = N;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    const int iter_copy_C = CEIL(block_M * block_N / kAccess, blockDim);

    half *D_tile = D + idx_block_M * block_M * ldc + idx_block_N * block_N;
    //HalfVector<8> buffer;
    half *shared_C = shared_mem_workspace;
    CSwizzle cSwizzle;

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < iter_copy_C; i++) {
        int idx = kAccess * (threadIdx.x + i * blockDim);
        half *src = shared_C + cSwizzle(idx);
        half *dst = D_tile + (idx / smem_ldc) * ldc + (idx % smem_ldc);

        //buffer.ld(src);
        //buffer.st(dst);
        *((float4*)dst) = *((float4*)src);
    }
    /* if(!blockIdx.x && !blockIdx.y && !threadIdx.x && !threadIdx.y){
        printf("OUT_32B\n");
    } */

// uncomment the line below to use 128-bit SMEM layout
#else
    const int kAccess = 8;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    const int lane = threadIdx.x & 31;
    int idx_warp_M  = (threadIdx.x >> 5) % (block_M / warp_M);

    const int iter_copy_C = CEIL(block_M * block_N / kAccess, blockDim*8); //8=mma_N

    __align__(16) uint out[16];

    half *D_tile = D + idx_block_M*block_M*N + idx_block_N*block_N + idx_warp_M*warp_M*N + ((lane>>3)<<2)*N + ((lane&7)<<3);

    half *shared_C = shared_mem_workspace + idx_warp_M * (warp_M>>5)*2368 + (lane&7)*296 + ((lane>>3)<<4); //16=8*2

    int ldc = N/2;

    #pragma unroll
    for(int m=0; m<iter_copy_C; m++){
        uint *dst = (uint*) (D_tile+m*32*N);
        half *src = (half*) (shared_C+m*2368);
        //for(int mi=0; mi<2; mi++){
        my_mma::ld_shared_float4(src,     &out[0]);
        my_mma::ld_shared_float4(src+8,   &out[1]);
        my_mma::ld_shared_float4(src+160, &out[2]);
        my_mma::ld_shared_float4(src+168, &out[3]);

        *(float4*)(dst)       = *(float4*)(&out[0]);
        *(float4*)(dst+ldc)   = *(float4*)(&out[4]);
        *(float4*)(dst+2*ldc) = *(float4*)(&out[8]);
        *(float4*)(dst+3*ldc) = *(float4*)(&out[12]);

        dst+=4*4*ldc;
        my_mma::ld_shared_float4(src+64,  &out[0]);
        my_mma::ld_shared_float4(src+72,  &out[1]);
        my_mma::ld_shared_float4(src+224, &out[2]);
        my_mma::ld_shared_float4(src+232, &out[3]);

        *(float4*)(dst)       = *(float4*)(&out[0]);
        *(float4*)(dst+ldc)   = *(float4*)(&out[4]);
        *(float4*)(dst+2*ldc) = *(float4*)(&out[8]);
        *(float4*)(dst+3*ldc) = *(float4*)(&out[12]);
        //}
    }
    /* if(!blockIdx.x && !blockIdx.y && !threadIdx.x && !threadIdx.y){
        printf("OUT_128B\n");
    } */
/*
    // uncomment the line below to use 32-bit SMEM layout

    const int kAccess = 8;
    const int smem_ldc = block_N;
    int ldc = N;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    const int iter_copy_C = CEIL(block_M * block_N / kAccess, blockDim);

    half *D_tile = D + idx_block_M * block_M * ldc + idx_block_N * block_N;
    //HalfVector<8> buffer;
    half *shared_C = shared_mem_workspace;
    CSwizzle cSwizzle;

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < iter_copy_C; i++) {
        int idx = kAccess * (threadIdx.x + i * blockDim);
        half *src = shared_C + cSwizzle(idx);
        half *dst = D_tile + (idx / smem_ldc) * ldc + (idx % smem_ldc);

        //buffer.ld(src);
        //buffer.st(dst);
        *((float4*)dst) = *((float4*)src);
    }
*/
#endif
}

// *** with row swizzling **
template<
    int block_M, int block_N, int blockDim,
    typename CSwizzle, bool GridMappingXYToMN
>
__device__ __forceinline__
void epilogue_impl
(const int M, const int N, const int *A_row_indices, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
    const int kAccess = 8;
    const int smem_ldc = block_N;
    half *shared_C = shared_mem_workspace;
    int ldc = N;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;

    int  *I_shared = (int*)(shared_C + block_M * block_N);
    int logical_row_start = idx_block_M * block_M;
    const int  *I_panel  = A_row_indices + logical_row_start;
    my_pipeline::copy_and_sync((unsigned*)I_shared, (const unsigned*)I_panel,
                                block_M);
    bool is_residue = (N % block_N != 0) &&
        (idx_block_N == (GridMappingXYToMN ? gridDim.y : gridDim.x) -1);

    half *D_tile = D + idx_block_N * block_N;
    HalfVector<8> buffer;
    CSwizzle cSwizzle;

    if (beta == 0.0f) {
        for (int i = 0; i < (CEIL(block_M * block_N / kAccess, blockDim)); i++) {
            int idx = kAccess * (threadIdx.x + i * blockDim);
            half *src = shared_C + cSwizzle(idx);
            bool valid = (idx < block_M * block_N);
            if (is_residue)
                valid = valid && ((idx % block_N)< (N-block_N*idx_block_N));
            if (valid) {
                int row_id = I_shared[idx / smem_ldc];
                half *dst = D_tile + row_id * ldc + (idx % smem_ldc);
                buffer.ld(src);
                buffer.mul(alpha);
                buffer.st(dst);
            }
        }
    }
    else {
        const half *C_tile = C + idx_block_N * block_N;
        HalfVector<8> buffer2;
        for (int i = 0; i < (CEIL(block_M * block_N / kAccess, blockDim)); i++) {
            int idx = kAccess * (threadIdx.x + i * blockDim);
            half *src = shared_C + cSwizzle(idx);
            bool valid = (idx < block_M * block_N);
            if (is_residue)
                valid = valid && ((idx % block_N)< (N-block_N*idx_block_N));
            if (valid) {
                int row_id = I_shared[idx / smem_ldc];
                int global_offset = row_id * ldc + (idx % smem_ldc);
                half *dst  = D_tile + global_offset;
                const half *src2 = C_tile + global_offset;
                buffer.ld(src);
                buffer2.ld(src2);
                buffer2.mul(beta);
                buffer.hfma(alpha, buffer2);
                buffer.st(dst);
            }
        }
    }
}

/*
template<
  int block_M, int block_N, int blockDim, typename ComputeType,
  typename OutputType, typename CSwizzle, bool GridMappingXYToMN
>
DEVICE_INLINE
void epilogue_impl(const int M, const int N, OutputType *D,
    const AccumulatorType* shared_mem_workspace,
    const float alpha, const OutputType* C, const float beta)
{
    const int kAccess = 128/(8*sizeof(ComputeType));
    ValueVector<ComputeType, kAccess> buffer;
    CSwizzle swizzle;

    int ldc = N;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    OutputType *D_tile = D + idx_block_M * block_M * ldc + idx_block_N * block_N;


    int iter = CEIL(block_M * block_M / kAccess, blockDim);
    int smem_ldc = block_N;
    if (beta == 0.0f) {
        for (int i = 0; i < iter; i++) {
            int idx = kAccess * (threadIdx.x + i*blockDim);
            const ComputeType* src = shared_mem_workspace + swizzle(idx);
            OutputType* dst = D_tile + (idx / smem_ldc) * ldc + (idx % smem_ldc);
            if (idx < block_M * block_N) {
                buffer.ld(src);
                buffer.mul(alpha);
                buffer.st(dst);
            }
        }
    }
    else {
        const OutputType *C_tile = C + idx_block_M * block_M * ldc
                                     + idx_block_N * block_N;
        ValueVector<OutputType, kAccess> buffer2;
        for (int i = 0; i < iter; i++) {
            int idx = kAccess * (threadIdx.x + i*blockDim);
            const ComputeType* src = shared_mem_workspace + swizzle(idx);
            int global_offset = (idx / smem_ldc) * ldc + (idx % smem_ldc);
            const OutputType* src2 = C_tile + global_offset
            OutputType* dst = D_tile + global_offset;
            if (idx < block_M * block_N) {
                buffer.ld(src);
                buffer2.ld(src2);
                buffer2.mul(beta);
                buffer.fma(alpha, buffer2);
                buffer.st(dst);
            }
        }
    }
}
*/

template<
    int block_M, int block_N, int blockDim, int warp_M, int warp_N, int mma_M, int mma_N, typename MmaShape, typename CSwizzle, bool GridMappingXYToMN
>
__device__ __forceinline__
void epilogue_impl
(const int M, const int N, half *D, half *shared_mem_workspace, float alpha, const half *C, const uint* C_metadata, float beta, const int nn, const int mm, const int brow, const int mbrow)
{
    const int warp_id = (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int kAccess = 8;

    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int idx_warp_M  = warp_id % (block_M / warp_M);
    int idx_warp_N  = warp_id / (block_M / warp_M);

    const int N_sp_p = ROUND_UP((ROUND_UP(N, mm)/mm)*nn, 16);
    const int mcol_n = N_sp_p/8;
    const int col_n  = N_sp_p*2;
    int mrow_m = 2;
    int bits_elem_meta=2;
    int nelems=(sizeof(uint)*8)/bits_elem_meta;
    int mcol_nn = nelems/mrow_m/nn; //8/n

    const half* C_panel = &C[idx_block_M*block_M*N_sp_p + idx_warp_M*warp_M*N_sp_p + idx_block_N*32*block_N + idx_warp_N*32*warp_N]; //32=4*8
    const uint* C_meta  = &C_metadata[idx_block_M*(block_M>>1)*mcol_n + idx_warp_M*(warp_M>>1)*mcol_n + (lane>>2)*2];

    const int smem_ldc = block_N;
    int ldc = N;

    const int iter_mma_M = warp_M / mma_M / 2;
    const int iter_mma_N = warp_N / mma_N;
    typedef typename my_mma::fragment_meta_sparse<MmaShape> FragmentMeta;
    FragmentMeta metafrag[iter_mma_M][iter_mma_N]; //={3642103998};

    #pragma unroll
    for(int m=0; m<iter_mma_M; m++){
        for(int n=0; n<iter_mma_N; n++){
            my_mma::load_meta_sync(metafrag[m][n].x, C_meta, m*16*mcol_n + n*16);
        }
    }
    //my_mma::load_meta_sync(metafrag[0][0].x, C_meta, 0);
    int index[4];
    for(int mbrow_iii=0; mbrow_iii<mrow_m; mbrow_iii++){
        //for(int mcol_ii=0; mcol_ii<mcol_nn; mcol_ii++){
            for (int n_i=0; n_i<nn; n_i++) {
                //cout << ((metadata[0][0].x >> (mbrow_iii*(nelems/mrow_m)*bits_elem_meta+mcol_ii*nn_row*bits_elem_meta+n_i*bits_elem_meta)) & 0x3);
                index[mbrow_iii*2+n_i] = ((metafrag[0][0].x[0] >> (
                            mbrow_iii*(nelems/mrow_m)*bits_elem_meta +
                            (lane%4)*nn*bits_elem_meta +
                            n_i*bits_elem_meta)) & 0x3);
            }
        //}
    }

    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0){
        printf("iter_copy_M:%d, iter_copy_N:%d \n", iter_mma_M, iter_mma_N);
        for(int ii=0; ii<4; ii++){
            printf("%u ", C_meta[ii]);
        } printf("\n");
        uint* data = metafrag[0][0].x;
        printf("metadata: ");
        printf("th%d ", threadIdx.x);
        for(int ii=0; ii<2; ii++){
            printf("%u ", data[ii]);
        }
        printf("\n");
        data = metafrag[0][1].x;
        printf("metadata: ");
        printf("th%d ", threadIdx.x);
        for(int ii=0; ii<2; ii++){
            printf("%u ", data[ii]);
        }
        printf("\n");
        printf("metadata: ");
        for(int ii=0; ii<4; ii++){
            printf("%d ", index[ii]);
        }
        printf("\n");
    }
}

}