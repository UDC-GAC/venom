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

#include "../common/base.h"
#include "../common/mma.h"
#include "../common/memcpy.h"
#include "../common/swizzle.h"
#include "../common/epilogue.h"

namespace spatha {

#ifdef BASELINE
template<
    // block-sparse pattern
    int BM_, int BK_,
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // threadblock level pipeline stage
    int      kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle,
    typename BSwizzle,
    typename CSwizzle,
    // pipeline configuration
    bool     UseRegisterDoubleBuffer = false,
    //bool     UseRegisterDoubleBuffer = true,
    bool     UseMinimumSync = true,
    bool     GridMappingXYToMN_ = true
>
struct SpmmBlockwiseKernel {
    static constexpr int BM      = BM_;
    static constexpr int BK      = BK_;
    static constexpr int block_M = ThreadBlockShape::M;
    static constexpr int block_N = ThreadBlockShape::N;
    static constexpr int block_K = ThreadBlockShape::K;
    static constexpr int warp_M  = WarpShape::M;
    static constexpr int warp_N  = WarpShape::N;
    static constexpr int mma_M   = MmaShape::M;
    static constexpr int mma_N   = MmaShape::N;
    static constexpr int mma_K   = MmaShape::K;
    /* static constexpr int brow    = 4;
    static constexpr int mbrow   = 16; */
    static_assert(BM == block_M,
    "Only support threadblock shape M == BM in block-sparse shape.\n");

    static_assert(WarpShape::K == ThreadBlockShape::K,
    "K-dim of warp and threadblock tiling must be the same. "
    "Split-K not supported.\n");

    static_assert( block_M % warp_M == 0);
    static_assert( block_N % warp_N == 0);
    static_assert( warp_M  % mma_M  == 0);
    static_assert( warp_N  % mma_N  == 0);
    static_assert( block_K % mma_K  == 0);
    static_assert( block_K % BK     == 0);
    //static_assert( kThreadBlockStage > 1);

    /* static_assert( warp_N % 16 == 0,
    "Only support warp shape M>=16 for performance.\n"); */

    static constexpr int metaPrefetchBlock = 512;
    static_assert(metaPrefetchBlock / block_K >= kThreadBlockStage);

    // precompute constants
    static constexpr bool GridMappingXYToMN = GridMappingXYToMN_;
    static constexpr int blockDim = 32 * (block_M/warp_M) * (block_N/warp_N);

    // use multi-stage shared-mem buffer (needed by async copy)
    static constexpr size_t input_buffer_size_static =
        (block_M*block_K/2 + block_N*block_K) * kThreadBlockStage * sizeof(half) + metaPrefetchBlock * 2 * sizeof(uint);
    /* static constexpr size_t input_buffer_size_static =
        (block_M + block_N) * block_K * kThreadBlockStage * sizeof(half)
        + metaPrefetchBlock * 2 * sizeof(int); */

    /* static constexpr size_t output_buffer_size_static =
        (block_M * block_N) * sizeof(half); */
    static constexpr size_t output_buffer_size_static =
        (block_M/(mma_M*2) * block_N/mma_N * (32+5) * 8) * sizeof(half);

    // mainloop interface
    __device__ __forceinline__
    void mainLoop(const int M, const int N, const int K,
        const int n, const int m, const int brow, const int mbrow,
        const unsigned int *A_metadata,
        const half *A_values, const uint* A_indices, const half *B, half *shared_mem_workspace
    );

    __device__ __forceinline__
    void epilogue(const int M, const int N, half *D,
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

    // row swizzling
    __device__ __forceinline__
    void epilogue(const int M, const int N, const int* A_row_indices, half *D,
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

};

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::mainLoop
(const int M, const int N, const int K,
    const int nn, const int mm, const int brow, const int mbrow,
    const unsigned int *A_metadata,
    const half *A_values, const uint *indices, const half *B, half *shared_mem_workspace)
{
    int warp_id = (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int idx_warp_M  = warp_id % (block_M / warp_M);
    int idx_warp_N  = warp_id / (block_M / warp_M);

    const half* A_panel = &A_values[idx_block_M*BM*(K>>1) + warp_id*16*(K>>1) + lane*8];
    const half* B_panel = &B[idx_block_N * block_N];
    const uint* M_panel = &A_metadata[idx_block_M*(block_M>>1)*(K>>4) + idx_warp_M*(warp_M>>1)*(K>>4) + (lane>>2)*2 + (lane&1)*16];

    ASwizzle aSwizzle;
    BSwizzle bSwizzle;

    // compute global to shared copy constants
    const int iter_copy_A = CEIL(block_M * 2, blockDim);
    const int iter_copy_B = CEIL(block_N * 4, blockDim);

    // compute shared memory buffer addresses
    const int NStage = kThreadBlockStage;
    const int size_of_tile_A = block_M * 16;
    const int size_of_tile_B = block_N * 32;
    half *shared_B = shared_mem_workspace;
    half *shared_A = shared_B + size_of_tile_B * NStage;

    // compute shared memory offsets
    int A_warp_panel_offset = idx_warp_M * warp_M*16;
    const int smem_lda = block_M;
    int B_warp_panel_offset = idx_warp_N * warp_N;
    const int smem_ldb = block_N;

    const int offset_A = A_warp_panel_offset + (lane/16)*256 + (lane%16)*8;
    const int iter_A_offset = (blockDim/32)*16*(K>>1);

    // define mma buffers
    typedef typename my_mma::fragment_a_sparse_cm<MmaShape> FragmentA;
    typedef typename my_mma::fragment_b_sparse_rm<MmaShape> FragmentB;
    typedef typename my_mma::fragment_c_sparse<MmaShape, AccumulatorType> FragmentC;
    typedef typename my_mma::fragment_meta_sparse<MmaShape> FragmentMeta;
    const int load_mma_M = warp_M / mma_M / 2;
    const int iter_mma_N = warp_N / mma_N;

    FragmentA afrag[load_mma_M];
    FragmentMeta metafrag[load_mma_M][2];//={3642103998};
    FragmentB bfrag[iter_mma_N];
    FragmentC cfrag[iter_mma_N][load_mma_M];

    // main loop
    const int num_tile = CEIL(K, 32);

    my_pipeline::Pipeline<NStage, true> pipe;
    int fetch = 0, compute = 0;

    for(; compute < num_tile; compute++) {

        const int index_I = compute*32;
        #pragma unroll
        for(int m=0; m<load_mma_M; m++){
            my_mma::load_meta_sync(metafrag[m][0].x, M_panel, index_I + m*16*(K>>4));
        }

        #pragma unroll
        for (; fetch < compute + NStage; fetch++) {
            pipe.acquire_writer();

            // fetch data
            if (fetch < num_tile) {
                const half* tile_A = A_panel + fetch * 256;
                half *shared_tile_A = shared_A + (fetch % NStage) * size_of_tile_A;
                half *shared_tile_B = shared_B + (fetch % NStage) * size_of_tile_B;

                // load A from GMEM to SMEM
                #pragma unroll
                for (int i = 0; i < iter_copy_A; i++) {
                    int idx = (threadIdx.x + blockDim*i) * 8;

                    const half *src = tile_A + i*iter_A_offset;
                          half *dst = shared_tile_A + aSwizzle(idx);
                    my_pipeline::cp_async_pred_zfill<16>(dst, src);
                }

                // load B from GMEM to SMEM
                #pragma unroll
                for (int i = 0; i < iter_copy_B; i++) {
                    int idx = (threadIdx.x + blockDim*i) * 8;
                    int nz_block_idx = (idx / block_N);
                    int k = nz_block_idx + fetch*32;

                    const half *src = B_panel + k * N + (idx % block_N);
                          half *dst = shared_tile_B + bSwizzle(idx);

                    my_pipeline::cp_async_pred_zfill<16>(dst, src);
                }

            }
            pipe.commit_stage();
        }
        pipe.acquire_reader();

        half *shared_tile_A = shared_A + (compute % NStage) * size_of_tile_A;
        half *shared_tile_B = shared_B + (compute % NStage) * size_of_tile_B;

        //#pragma unroll
        #pragma unroll
        for (int m = 0; m < load_mma_M; m++) {
            my_mma::lds_matrix_sync<ASwizzle>(afrag[m].x, shared_tile_A, offset_A+(m<<5)*16);
        }

        #pragma unroll
        for (int n = 0; n < iter_mma_N; n++) {
            my_mma::load_matrix_sync<BSwizzle>(bfrag[n], shared_tile_B, B_warp_panel_offset + n*mma_N, smem_ldb);
        }

        #pragma unroll
        for (int mi = 0; mi < load_mma_M; mi++) {
            #pragma unroll
            for (int n = 0; n < iter_mma_N; n++) {
                mma_sync_sparse_v(cfrag[n][mi], afrag[mi], bfrag[n], cfrag[n][mi], metafrag[mi][0]);
            }
        }
        pipe.release_reader();
    }

#ifdef OUT_32B
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int row  = lane / 4;
    int col  = (lane & 3) *2;
    int C_warp_tile_offset = idx_warp_M * warp_M * smem_ldc + idx_warp_N * warp_N + row*smem_ldc*4 + col;
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < load_mma_M; mi++) {
        int offset = C_warp_tile_offset + mi*32*smem_ldc;
        #pragma unroll
        for (int n = 0; n <iter_mma_N; n++) {
            my_mma::store_matrix_sync<CSwizzle>(cfrag[n][mi], shared_C, offset+(n)*mma_N, smem_ldc);
        }
        //__threadfence_block();
    }
#else
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int C_warp_tile_offset = idx_warp_M*(warp_M>>5)*2368 + ((lane>>1)&1)*160 + ((lane>>2)<<4) + ((lane&1)<<3);
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < load_mma_M; mi++) {
        int offset = C_warp_tile_offset + mi*2368; //(32+5)*8*8
        #pragma unroll
        for (int n = 0; n <iter_mma_N; n++) {
            my_mma::store_matrix_sync_128<CSwizzle>(cfrag[n][mi], shared_C, offset+n*296, smem_ldc); // 296=(32+5)*8
        }
    }

#endif
}

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
) {
    epilogue_impl<BM, block_N, blockDim, warp_M, CSwizzle, GridMappingXYToMN>(M, N, D,
        shared_mem_workspace, alpha, C, beta);
}

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, const int *A_row_indices, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
    epilogue_impl<BM, block_N, blockDim, warp_M, CSwizzle, GridMappingXYToMN>(M, N,
        A_row_indices, D, shared_mem_workspace, alpha, C, beta);
}

#elif IDEAL_KERNEL
template<
    // block-sparse pattern
    int BM_, int BK_,
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // threadblock level pipeline stage
    int      kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle,
    typename BSwizzle,
    typename CSwizzle,
    // pipeline configuration
    bool     UseRegisterDoubleBuffer = false,
    //bool     UseRegisterDoubleBuffer = true,
    bool     UseMinimumSync = true,
    bool     GridMappingXYToMN_ = true
>
struct SpmmBlockwiseKernel {
    static constexpr int BM      = BM_;
    static constexpr int BK      = BK_;
    static constexpr int block_M = ThreadBlockShape::M;
    static constexpr int block_N = ThreadBlockShape::N;
    static constexpr int block_K = ThreadBlockShape::K;
    static constexpr int warp_M  = WarpShape::M;
    static constexpr int warp_N  = WarpShape::N;
    static constexpr int mma_M   = MmaShape::M;
    static constexpr int mma_N   = MmaShape::N;
    static constexpr int mma_K   = MmaShape::K;
    /* static constexpr int brow    = 4;
    static constexpr int mbrow   = 16; */
    static_assert(BM == block_M,
    "Only support threadblock shape M == BM in block-sparse shape.\n");

    static_assert(WarpShape::K == ThreadBlockShape::K,
    "K-dim of warp and threadblock tiling must be the same. "
    "Split-K not supported.\n");

    static_assert( block_M % warp_M == 0);
    static_assert( block_N % warp_N == 0);
    static_assert( warp_M  % mma_M  == 0);
    static_assert( warp_N  % mma_N  == 0);
    static_assert( block_K % mma_K  == 0);
    static_assert( block_K % BK     == 0);
    //static_assert( kThreadBlockStage > 1);

    /* static_assert( warp_N % 16 == 0,
    "Only support warp shape M>=16 for performance.\n"); */

    static constexpr int metaPrefetchBlock = 512; //4 warps*4
    static_assert(metaPrefetchBlock / (block_K) >= kThreadBlockStage);

    // precompute constants
    static constexpr bool GridMappingXYToMN = GridMappingXYToMN_;
    static constexpr int blockDim = 32 * (block_M/warp_M) * (block_N/warp_N);

    // use multi-stage shared-mem buffer (needed by async copy)
    static constexpr size_t input_buffer_size_static =
        (block_M*block_K/2 + block_N*block_K) * kThreadBlockStage * sizeof(half) + metaPrefetchBlock * 2 * sizeof(uint);
    /* static constexpr size_t input_buffer_size_static =
        (block_M + block_N) * block_K * kThreadBlockStage * sizeof(half)
        + metaPrefetchBlock * 2 * sizeof(int); */

    /* static constexpr size_t output_buffer_size_static =
        (block_M * block_N) * sizeof(half); */
    static constexpr size_t output_buffer_size_static =
        (block_M/(mma_M*2) * block_N/mma_N * (32+5) * 8) * sizeof(half);
    /* static constexpr size_t output_buffer_size_static =
        (block_M/(mma_M*2)/(mma_M*2) * block_N/mma_N * (32+5) * 8/mma_N * (32+5) * 8) * sizeof(half); */

    // mainloop interface
    __device__ __forceinline__
    void mainLoop(const int M, const int N, const int K,
        const int n, const int m, const int brow, const int mbrow,
        const unsigned int *A_metadata,
        const half *A_values, const uint* A_indices, const half *B, half *shared_mem_workspace
    );

    __device__ __forceinline__
    void epilogue(const int M, const int N, half *D,
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

    // row swizzling
    __device__ __forceinline__
    void epilogue(const int M, const int N, const int* A_row_indices, half *D,
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

};

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::mainLoop
(const int M, const int N, const int K,
    const int nn, const int mm, const int brow, const int mbrow,
    const unsigned int *A_metadata,
    const half *A_values, const uint *A_indices, const half *B, half *shared_mem_workspace)
{
    const int warp_id = (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    //const int K_sp = (ROUND_UP(K, mm)/mm)*nn;
    const int K_sp_p = ROUND_UP((ROUND_UP(K, mm)/mm)*nn, 16);
    const int mcol_k = K_sp_p/8; //K_sp_p/nn/mcol_kk;
    const int col_k = K_sp_p*2;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int idx_warp_M  = warp_id % (block_M / warp_M);
    int idx_warp_N  = warp_id / (block_M / warp_M);

    const half* A_panel = &A_values[idx_block_M*BM*K_sp_p + warp_id*16*K_sp_p + lane*8];
    const half* B_panel = &B[idx_block_N * block_N];
    const uint* M_panel = &A_metadata[idx_block_M*(block_M>>1)*mcol_k + idx_warp_M*(warp_M>>1)*mcol_k + (lane>>2)*2 + (lane&1)*16];
    const uint* I_panel = &A_indices[idx_block_M*col_k];

    ASwizzle aSwizzle;
    BSwizzle bSwizzle;

    // compute global to shared copy constants
    const int iter_copy_A = CEIL(block_M * 2, blockDim);
    const int iter_copy_B = CEIL(block_N * 4, blockDim); //block_K/kAccess

    // compute shared memory buffer addresses
    const int NStage = kThreadBlockStage;
    const int size_of_tile_A = block_M * 16; //block_K/2
    const int size_of_tile_B = block_N * 32;
    half *shared_B = shared_mem_workspace;
    half *shared_A = shared_B + size_of_tile_B * NStage;
    //uint  *shared_I = (uint*)(shared_A + size_of_tile_A * NStage);

    // compute shared memory offsets
    int A_warp_panel_offset = idx_warp_M * warp_M*16; //block_K/2
    const int smem_lda = block_M;
    int B_warp_panel_offset = idx_warp_N * warp_N;
    const int smem_ldb = block_N;

    const int offset_A = A_warp_panel_offset + (lane/16)*256 + (lane%16)*8;
    const int iter_A_offset = (blockDim/32)*16*K_sp_p;

    // define mma buffers
    typedef typename my_mma::fragment_a_sparse_cm<MmaShape> FragmentA;
    typedef typename my_mma::fragment_b_sparse_rm<MmaShape> FragmentB;
    typedef typename my_mma::fragment_c_sparse<MmaShape, AccumulatorType> FragmentC;
    typedef typename my_mma::fragment_meta_sparse<MmaShape> FragmentMeta;
    const int load_mma_M = warp_M / mma_M / 2;
    const int iter_mma_N = warp_N / mma_N;

    FragmentMeta metafrag[load_mma_M][2];//={3642103998};
    FragmentA afrag[load_mma_M];
    FragmentB bfrag[iter_mma_N];
    FragmentC cfrag[iter_mma_N][load_mma_M];

    // main loop
    const int num_tile = CEIL(K_sp_p*2, block_K);

    my_pipeline::Pipeline<NStage, true> pipe;
    int fetch = 0, compute = 0;

    #pragma unroll
    for(; compute < num_tile; compute++) {

        const int index_I = compute*32;
        #pragma unroll
        for(int m=0; m<load_mma_M; m++){
            my_mma::load_meta_sync(metafrag[m][0].x, M_panel, index_I + m*16*mcol_k);
        }

        #pragma unroll
        for (; fetch < compute + NStage; fetch++) {
            pipe.acquire_writer();

            // prefetch metadata

            // fetch data
            if (fetch < num_tile) {
                const half* tile_A = A_panel + fetch * 256;
                half *shared_tile_A = shared_A + (fetch % NStage) * size_of_tile_A;
                half *shared_tile_B = shared_B + (fetch % NStage) * size_of_tile_B;

                // load A from GMEM to SMEM
                #pragma unroll
                for (int i = 0; i < iter_copy_A; i++) {
                    int idx = (threadIdx.x + blockDim*i) * 8;

                    const half *src = tile_A + i*iter_A_offset;
                          half *dst = shared_tile_A + aSwizzle(idx);

                    my_pipeline::cp_async_pred_zfill<16>(dst, src);
                }

                // load B from GMEM to SMEM
                #pragma unroll
                for (int i = 0; i < iter_copy_B; i++) {
                    int idx = (threadIdx.x + blockDim*i) * 8;
                    int nz_block_idx = (idx / block_N);

                    //int k_base = tile_I[ nz_block_idx ];
                    int k_base = nz_block_idx%4;
                    int k = (nz_block_idx/4)*mm + fetch*8*mm + k_base;
                    bool zfill = (k>K);

                    const half *src = B_panel + k * N + (idx % block_N);
                          half *dst = shared_tile_B + bSwizzle(idx);

                    my_pipeline::cp_async_pred_zfill<16>(dst, src, true, zfill);
                }

            }
            pipe.commit_stage();
        }
        pipe.acquire_reader();

        half *shared_tile_A = shared_A + (compute % NStage) * size_of_tile_A;
        half *shared_tile_B = shared_B + (compute % NStage) * size_of_tile_B;

        //#pragma unroll
        #pragma unroll
        for (int m = 0; m < load_mma_M; m++) {
            my_mma::lds_matrix_sync<ASwizzle>(afrag[m].x, shared_tile_A, offset_A+(m<<5)*16);
        }

        #pragma unroll
        for (int n = 0; n < iter_mma_N; n++) {
            my_mma::load_matrix_sync<BSwizzle>(bfrag[n], shared_tile_B, B_warp_panel_offset + n*mma_N, smem_ldb);
        }

        #pragma unroll
        for (int mi = 0; mi < load_mma_M; mi++) {
            #pragma unroll
            for (int n = 0; n < iter_mma_N; n++) {
                mma_sync_sparse_v(cfrag[n][mi], afrag[mi], bfrag[n], cfrag[n][mi], metafrag[mi][0]);
            }
        }
        pipe.release_reader();
    }

#ifdef OUT_32B
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int row  = lane / 4;
    int col  = (lane & 3) *2;
    int C_warp_tile_offset = idx_warp_M * warp_M * smem_ldc + idx_warp_N * warp_N + row*smem_ldc*4 + col;
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < load_mma_M; mi++) {
        int offset = C_warp_tile_offset + mi*32*smem_ldc;
        #pragma unroll
        for (int n = 0; n <iter_mma_N; n++) {
            my_mma::store_matrix_sync<CSwizzle>(cfrag[n][mi], shared_C, offset+(n)*mma_N, smem_ldc);
        }
        //__threadfence_block();
    }
#else
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int C_warp_tile_offset = idx_warp_M*(warp_M>>5)*2368 + ((lane>>1)&1)*160 + ((lane>>2)<<4) + ((lane&1)<<3);
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < load_mma_M; mi++) {
        int offset = C_warp_tile_offset + mi*2368; //(32+5)*8*8
        #pragma unroll
        for (int n = 0; n <iter_mma_N; n++) {
            my_mma::store_matrix_sync_128<CSwizzle>(cfrag[n][mi], shared_C, offset+n*296, smem_ldc); // 296=(32+5)*8
        }
    }
#endif
}

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
) {
    epilogue_impl<BM, block_N, blockDim, warp_M, CSwizzle, GridMappingXYToMN>(M, N, D,
        shared_mem_workspace, alpha, C, beta);
}

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, const int *A_row_indices, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
    epilogue_impl<BM, block_N, blockDim, warp_M, CSwizzle, GridMappingXYToMN>(M, N,
        A_row_indices, D, shared_mem_workspace, alpha, C, beta);
}

#else
template<
    // block-sparse pattern
    int BM_, int BK_,
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // threadblock level pipeline stage
    int      kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle,
    typename BSwizzle,
    typename CSwizzle,
    // pipeline configuration
    bool     UseRegisterDoubleBuffer = false,
    //bool     UseRegisterDoubleBuffer = true,
    bool     UseMinimumSync = true,
    bool     GridMappingXYToMN_ = true
>
struct SpmmBlockwiseKernel {
    static constexpr int BM      = BM_;
    static constexpr int BK      = BK_;
    static constexpr int block_M = ThreadBlockShape::M;
    static constexpr int block_N = ThreadBlockShape::N;
    static constexpr int block_K = ThreadBlockShape::K;
    static constexpr int warp_M  = WarpShape::M;
    static constexpr int warp_N  = WarpShape::N;
    static constexpr int mma_M   = MmaShape::M;
    static constexpr int mma_N   = MmaShape::N;
    static constexpr int mma_K   = MmaShape::K;
    /* static constexpr int brow    = 4;
    static constexpr int mbrow   = 16; */
    static_assert(BM == block_M,
    "Only support threadblock shape M == BM in block-sparse shape.\n");

    static_assert(WarpShape::K == ThreadBlockShape::K,
    "K-dim of warp and threadblock tiling must be the same. "
    "Split-K not supported.\n");

    static_assert( block_M % warp_M == 0);
    static_assert( block_N % warp_N == 0);
    static_assert( warp_M  % mma_M  == 0);
    static_assert( warp_N  % mma_N  == 0);
    static_assert( block_K % mma_K  == 0);
    static_assert( block_K % BK     == 0);
    //static_assert( kThreadBlockStage > 1);

    /* static_assert( warp_N % 16 == 0,
    "Only support warp shape M>=16 for performance.\n"); */

    static constexpr int metaPrefetchBlock = 512; //4 warps*4
    static_assert(metaPrefetchBlock / (block_K) >= kThreadBlockStage);

    // precompute constants
    static constexpr bool GridMappingXYToMN = GridMappingXYToMN_;
    static constexpr int blockDim = 32 * (block_M/warp_M) * (block_N/warp_N);

    // use multi-stage shared-mem buffer (needed by async copy)
    static constexpr size_t input_buffer_size_static =
        (block_M*block_K/2 + block_N*block_K) * kThreadBlockStage * sizeof(half) + metaPrefetchBlock * 2 * sizeof(uint);
    /* static constexpr size_t input_buffer_size_static =
        (block_M + block_N) * block_K * kThreadBlockStage * sizeof(half)
        + metaPrefetchBlock * 2 * sizeof(int); */

    /* static constexpr size_t output_buffer_size_static =
        (block_M * block_N) * sizeof(half); */
    static constexpr size_t output_buffer_size_static =
        (block_M/(mma_M*2) * block_N/mma_N * (32+5) * 8) * sizeof(half);
    /* static constexpr size_t output_buffer_size_static =
        (block_M/(mma_M*2)/(mma_M*2) * block_N/mma_N * (32+5) * 8/mma_N * (32+5) * 8) * sizeof(half); */

    // mainloop interface
    __device__ __forceinline__
    void mainLoop(const int M, const int N, const int K,
        const int n, const int m, const int brow, const int mbrow,
        const unsigned int *A_metadata,
        const half *A_values, const uint* A_indices, const half *B, half *shared_mem_workspace
    );

    __device__ __forceinline__
    void epilogue(const int M, const int N, half *D,
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

    // row swizzling
    __device__ __forceinline__
    void epilogue(const int M, const int N, const int* A_row_indices, half *D,
        half *shared_mem_workspace, float alpha, const half *C, float beta
    );

};

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::mainLoop
(const int M, const int N, const int K,
    const int nn, const int mm, const int brow, const int mbrow,
    const unsigned int *A_metadata,
    const half *A_values, const uint *A_indices, const half *B, half *shared_mem_workspace)
{
    const int warp_id = (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int K_sp_p = ROUND_UP((ROUND_UP(K, mm)/mm)*nn, 16);
    const int mcol_k = K_sp_p/8; //K_sp_p/nn/mcol_kk;
    const int col_k = K_sp_p*2;
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int idx_warp_M  = warp_id % (block_M / warp_M);
    int idx_warp_N  = warp_id / (block_M / warp_M);

    const half* A_panel = &A_values[idx_block_M*BM*K_sp_p + warp_id*16*K_sp_p + lane*8];
    const half* B_panel = &B[idx_block_N * block_N];
    const uint* M_panel = &A_metadata[idx_block_M*(block_M>>1)*mcol_k + idx_warp_M*(warp_M>>1)*mcol_k + (lane>>2)*2 + (lane&1)*16];
    const uint* I_panel = &A_indices[idx_block_M*col_k];

    ASwizzle aSwizzle;
    BSwizzle bSwizzle;

    // compute global to shared copy constants
    const int iter_copy_A = CEIL(block_M * 2, blockDim);
    const int iter_copy_B = CEIL(block_N * 4, blockDim); //block_K/kAccess

    // compute shared memory buffer addresses
    const int NStage = kThreadBlockStage;
    const int size_of_tile_A = block_M * 16; //block_K/2
    const int size_of_tile_B = block_N * 32;
    half *shared_B = shared_mem_workspace;
    half *shared_A = shared_B + size_of_tile_B * NStage;
    uint  *shared_I = (uint*)(shared_A + size_of_tile_A * NStage);

    // compute shared memory offsets
    int A_warp_panel_offset = idx_warp_M * warp_M*16; //block_K/2
    const int smem_lda = block_M;
    int B_warp_panel_offset = idx_warp_N * warp_N;
    const int smem_ldb = block_N;

    const int offset_A = A_warp_panel_offset + (lane/16)*256 + (lane%16)*8;
    const int iter_A_offset = (blockDim/32)*16*K_sp_p;

    // define mma buffers
    typedef typename my_mma::fragment_a_sparse_cm<MmaShape> FragmentA;
    typedef typename my_mma::fragment_b_sparse_rm<MmaShape> FragmentB;
    typedef typename my_mma::fragment_c_sparse<MmaShape, AccumulatorType> FragmentC;
    typedef typename my_mma::fragment_meta_sparse<MmaShape> FragmentMeta;
    const int load_mma_M = warp_M / mma_M / 2;
    const int iter_mma_N = warp_N / mma_N;

    FragmentMeta metafrag[load_mma_M][2];//={3642103998};
    FragmentA afrag[load_mma_M];
    FragmentB bfrag[iter_mma_N];
    FragmentC cfrag[iter_mma_N][load_mma_M];

    // main loop
    const int num_tile = CEIL(K_sp_p*2, block_K);

    my_pipeline::Pipeline<NStage, true> pipe;
    int fetch = 0, compute = 0;

    int num_block = col_k;
    int num_meta_tile = CEIL(num_block, metaPrefetchBlock);
    int fetch_meta = 0;
    const int prefetchMetaStage = metaPrefetchBlock / block_K; //(block_K/4)
    // load the first tile of indices
    my_pipeline::copy_and_sync((uint*)shared_I, (const uint*)I_panel,
                                min(num_block, metaPrefetchBlock));
    fetch_meta++;

    #pragma unroll
    for(; compute < num_tile; compute++) {

        const int index_I = compute*32;
        #pragma unroll
        for(int m=0; m<load_mma_M; m++){
            my_mma::load_meta_sync(metafrag[m][0].x, M_panel, index_I + m*16*mcol_k);
        }

        #pragma unroll
        for (; fetch < compute + NStage; fetch++) {
            pipe.acquire_writer();

            // prefetch metadata
            if (fetch % prefetchMetaStage ==0) {
                if(fetch_meta < num_meta_tile) {
                    uint *shared_tile_I = shared_I + (fetch_meta %2)*metaPrefetchBlock;
                    const uint *tile_I = I_panel + fetch_meta * metaPrefetchBlock;
                    int meta_num = min(metaPrefetchBlock,
                                        num_block - fetch_meta * metaPrefetchBlock);

                    my_pipeline::cp_async_tile<metaPrefetchBlock, blockDim>(
                        (uint*)shared_tile_I, (const uint*)tile_I, meta_num);
                }
                fetch_meta++;
            }

            // fetch data
            if (fetch < num_tile) {
                const half* tile_A = A_panel + fetch * 256;
                uint *shared_tile_I = shared_I + (fetch_meta %2)*metaPrefetchBlock;
                uint *tile_I = shared_tile_I + (fetch % prefetchMetaStage) * block_K; // block_K/4
                half *shared_tile_A = shared_A + (fetch % NStage) * size_of_tile_A;
                half *shared_tile_B = shared_B + (fetch % NStage) * size_of_tile_B;

                // load A from GMEM to SMEM
                #pragma unroll
                for (int i = 0; i < iter_copy_A; i++) {
                    int idx = (threadIdx.x + blockDim*i) * 8;

                    const half *src = tile_A + i*iter_A_offset;
                          half *dst = shared_tile_A + aSwizzle(idx);

                    my_pipeline::cp_async_pred_zfill<16>(dst, src);
                }

                // load B from GMEM to SMEM
                #pragma unroll
                for (int i = 0; i < iter_copy_B; i++) {
                    int idx = (threadIdx.x + blockDim*i) * 8;
                    int nz_block_idx = (idx / block_N);

                    int k_base = tile_I[ nz_block_idx ];
                    int k = (nz_block_idx/4)*mm + fetch*8*mm + k_base;
                    bool zfill = (k>K);

                    const half *src = B_panel + k * N + (idx % block_N);
                          half *dst = shared_tile_B + bSwizzle(idx);

                    my_pipeline::cp_async_pred_zfill<16>(dst, src, true, zfill);
                }

            }
            pipe.commit_stage();
        }
        pipe.acquire_reader();

        half *shared_tile_A = shared_A + (compute % NStage) * size_of_tile_A;
        half *shared_tile_B = shared_B + (compute % NStage) * size_of_tile_B;

        //#pragma unroll
        #pragma unroll
        for (int m = 0; m < load_mma_M; m++) {
            my_mma::lds_matrix_sync<ASwizzle>(afrag[m].x, shared_tile_A, offset_A+(m<<5)*16);
        }

        #pragma unroll
        for (int n = 0; n < iter_mma_N; n++) {
            my_mma::load_matrix_sync<BSwizzle>(bfrag[n], shared_tile_B, B_warp_panel_offset + n*mma_N, smem_ldb);
        }

        #pragma unroll
        for (int mi = 0; mi < load_mma_M; mi++) {
            #pragma unroll
            for (int n = 0; n < iter_mma_N; n++) {
                mma_sync_sparse_v(cfrag[n][mi], afrag[mi], bfrag[n], cfrag[n][mi], metafrag[mi][0]);
            }
        }
        pipe.release_reader();
    }
#ifdef OUT_32B
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int row  = lane / 4;
    int col  = (lane & 3) *2;
    int C_warp_tile_offset = idx_warp_M * warp_M * smem_ldc + idx_warp_N * warp_N + row*smem_ldc*4 + col;
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < load_mma_M; mi++) {
        int offset = C_warp_tile_offset + mi*32*smem_ldc;
        #pragma unroll
        for (int n = 0; n <iter_mma_N; n++) {
            my_mma::store_matrix_sync<CSwizzle>(cfrag[n][mi], shared_C, offset+(n)*mma_N, smem_ldc);
        }
        //__threadfence_block();
    }
#else
    //uncomment the lines below to use STS.128 bit
    half *shared_C = shared_mem_workspace;
    const int smem_ldc = block_N;
    int C_warp_tile_offset = idx_warp_M*(warp_M>>5)*2368 + ((lane>>1)&1)*160 + ((lane>>2)<<4) + ((lane&1)<<3);
    __syncthreads();

    #pragma unroll
    for (int mi = 0; mi < load_mma_M; mi++) {
        int offset = C_warp_tile_offset + mi*2368; //(32+5)*8*8
        #pragma unroll
        for (int n = 0; n <iter_mma_N; n++) {
            my_mma::store_matrix_sync_128<CSwizzle>(cfrag[n][mi], shared_C, offset+n*296, smem_ldc); // 296=(32+5)*8
        }
    }
#endif
}

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
) {
    epilogue_impl<BM, block_N, blockDim, warp_M, CSwizzle, GridMappingXYToMN>(M, N, D,
        shared_mem_workspace, alpha, C, beta);
}

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SpmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, const int *A_row_indices, half *D,
    half *shared_mem_workspace, float alpha, const half *C, float beta
)
{
    epilogue_impl<BM, block_N, blockDim, warp_M, CSwizzle, GridMappingXYToMN>(M, N,
        A_row_indices, D, shared_mem_workspace, alpha, C, beta);
}

#endif

}
