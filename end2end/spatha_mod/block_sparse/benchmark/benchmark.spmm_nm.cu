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


#include "../spmm/spmm_op.h"
#include "../spmm/spmm_library_decl.h"
#include "../cuda_array.h"
#include "argparse_util.h"
#include "timing_util.h"

using namespace spatha;

inline float benchmark(SpmmNMInitFn_t init_fn,
    SpmmNMExecFn_t exec_fn,
    BlockwiseSpMatrix<half> &spmat, int N, half *B, half *D,
    cudaStream_t stream = NULL, int warmup=10, int repeat = 100)
{
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return -1;
    }

    GpuTimer gpu_timer;

    SpmmBlockwiseOpState state = (*init_fn)(spmat, N, B, D);
    if (!state.initSuccess) {
        std::cerr << "return due to unsuccessful initialization. " << std::endl;
        return -1;
    }

    (*exec_fn)(state, stream);
    cudaDeviceSynchronize();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "kernel failed." << std::endl;
        return -1;
    }

    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            gpu_timer.start();

        (*exec_fn)(state, stream);
    }
    gpu_timer.stop();


    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "kernel failed." << std::endl;
        return -1;
    }
    float dur = gpu_timer.elapsed_msecs() / repeat;
    return dur;
}

int main(int argc, const char** argv)
{
    int m, n, k;
    int pattern_code, block_sz, meta_block_sz;
    float density;
    unsigned seed;
    parseSpmmArgs(argc, argv, m, n, k, density, seed, pattern_code,
    block_sz, meta_block_sz, /* verbose */false);

    // pad shapes
    const int n_pad_to = 32;
    /* if (m % block_sz != 0) {
        if (load_pattern) {
            std::cerr << "Loaded matrix shape nrow is not padded to block_sz\n";
            exit(EXIT_FAILURE);
        }
        else {
            m += (block_sz - (m % block_sz));
            std::cerr << "m padded to : " << m << "\n";
        }
    } */
    if (n % n_pad_to != 0) {
        n += (n_pad_to - (n % n_pad_to));
        std::cerr << "n padded to : " << n << " for better alignment.\n";
    }

    CudaRandomArray<half> B;
    CudaZerosArray<half> D;
    B.initialize(k*n);
    D.initialize(m*n);
    B.sync_device();
    D.sync_device();

    // branch on pattern_code
    if (pattern_code == 0) { // block-2in4

        spatha::BlockwiseSpMatrix<half> spmat;

        //spmat.init_sparse(m, k, block_sz, 1, density, row_permute, seed);
        const int swizzle = 4;
        spmat.init_random(m, k, swizzle, 2, 4, density, seed, meta_block_sz);

        spmat.transform_and_sync_device();

        // benchmark
        float gflop_count = (float)m / 1e9 * n*k*2;

#define BENCHMARK(BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE) \
{\
    std::cout << GPU_CC << "," << spmat.config_str << ",";\
    printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", n, BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE);\
    float dur = benchmark( \
            NAME_FUNC(SpmmNM, Init, BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE), \
            NAME_FUNC(SpmmNM, Exec, BLOCK_SZ, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE), \
            spmat, n, B.device_ptr, D.device_ptr); \
    printf("%f,%f\n", dur*1e3, gflop_count/dur); \
}

        switch (block_sz) {
            case 32:
            BENCHMARK(32, 16, 16, 32, 16, 16, 16, 8, 16, 2);

            BENCHMARK(32, 64, 16, 32, 64, 16, 16, 8, 16, 2);
            BENCHMARK(32, 64, 16, 32, 64, 16, 16, 8, 16, 3);
            BENCHMARK(32, 64, 16, 32, 64, 16, 16, 8, 16, 4);

            BENCHMARK(32, 64, 128, 32, 64, 128, 16, 8, 16, 2);

            BENCHMARK(32, 128, 16, 32, 128, 16, 16, 8, 16, 2);
            BENCHMARK(32, 128, 16, 32, 64, 16, 16, 8, 16, 2);
            BENCHMARK(32, 128, 16, 32, 32, 16, 16, 8, 16, 2);
            BENCHMARK(32, 128, 16, 32, 16, 16, 16, 8, 16, 2);

            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 8, 16, 2);
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 8, 16, 3);
            BENCHMARK(32, 64, 16, 32, 16, 16, 16, 8, 16, 4);

            /* BENCHMARK(32, 64, 64, 32, 32, 64, 16, 8, 32, 1);
            BENCHMARK(32, 64, 64, 32, 32, 64, 16, 8, 16, 1);
            BENCHMARK(32, 64, 32, 32, 32, 32, 16, 8, 32, 1);
            BENCHMARK(32, 32, 32, 32, 32, 32, 16, 8, 32, 1);
            BENCHMARK(32, 32, 32, 32, 32, 32, 16, 8, 16, 1);
            BENCHMARK(32, 64, 64, 32, 64, 64, 16, 8, 32, 1); */

            BENCHMARK(32, 64, 64, 32, 32, 64, 16, 8, 32, 2);
            BENCHMARK(32, 64, 64, 32, 32, 64, 16, 8, 32, 3);
            BENCHMARK(32, 64, 64, 32, 32, 64, 16, 8, 32, 4);

            BENCHMARK(32, 32, 32, 32, 32, 32, 16, 8, 32, 2);

            BENCHMARK(32, 32, 32, 32, 32, 32, 16, 8, 16, 2);

            BENCHMARK(32, 64, 32, 32, 32, 32, 16, 8, 16, 2);

            BENCHMARK(32, 64, 32, 32, 64, 32, 16, 8, 16, 2);

            BENCHMARK(32, 128, 64, 32, 32, 64, 16, 8, 16, 2);

            //BENCHMARK(32, 128, 64, 32, 32, 64, 16, 8, 32, 1);
            BENCHMARK(32, 128, 64, 32, 32, 64, 16, 8, 32, 2);
            BENCHMARK(32, 128, 64, 32, 32, 64, 16, 8, 32, 3);
            BENCHMARK(32, 128, 64, 32, 32, 64, 16, 8, 32, 4);

            BENCHMARK(32, 64, 64, 32, 16, 64, 16, 8, 32, 2);
            BENCHMARK(32, 64, 64, 32, 16, 64, 16, 8, 32, 3);
            BENCHMARK(32, 64, 64, 32, 16, 64, 16, 8, 32, 4);

            BENCHMARK(32, 64, 64, 32, 64, 64, 16, 8, 32, 2);
            BENCHMARK(32, 64, 64, 32, 64, 64, 16, 8, 32, 3);
            BENCHMARK(32, 64, 64, 32, 64, 64, 16, 8, 32, 4);

            BENCHMARK(32, 64, 32, 32, 64, 32, 16, 8, 32, 2);

            BENCHMARK(32, 64, 128, 32, 64, 128, 16, 8, 32, 2);

            BENCHMARK(32, 128, 128, 32, 64, 128, 16, 8, 32, 2);

            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 8, 32, 2);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 8, 32, 3);
            BENCHMARK(32, 64, 32, 32, 16, 32, 16, 8, 32, 4);

            //BENCHMARK(32, 256, 32, 32, 64, 32, 16, 8, 32, 2);

            BENCHMARK(32, 128, 64, 32, 64, 64, 16, 8, 32, 2);
            BENCHMARK(32, 128, 64, 32, 64, 64, 16, 8, 32, 3);
            BENCHMARK(32, 128, 64, 32, 64, 64, 16, 8, 32, 4);

            BENCHMARK(32, 128, 128, 32, 128, 128, 16, 8, 32, 2);
            /* BENCHMARK(32, 128, 128, 32, 128, 128, 16, 8, 32, 3);
            BENCHMARK(32, 128, 128, 32, 128, 128, 16, 8, 32, 4); */
            break;

            case 64:
            BENCHMARK(64, 32, 32, 64, 32, 32, 16, 8, 32, 2);
            BENCHMARK(64, 32, 32, 64, 32, 32, 16, 8, 32, 3);
            BENCHMARK(64, 32, 32, 64, 32, 32, 16, 8, 32, 4);

            BENCHMARK(64, 64, 16, 64, 32, 16, 16, 8, 16, 2);
            BENCHMARK(64, 64, 16, 64, 32, 16, 16, 8, 16, 3);
            BENCHMARK(64, 64, 16, 64, 32, 16, 16, 8, 16, 4);

            BENCHMARK(64, 64, 32, 64, 32, 32, 16, 8, 32, 2);
            BENCHMARK(64, 64, 32, 64, 32, 32, 16, 8, 32, 3);
            BENCHMARK(64, 64, 32, 64, 32, 32, 16, 8, 32, 4);

            BENCHMARK(64, 64, 32, 64, 32, 32, 16, 8, 16, 2);
            BENCHMARK(64, 64, 32, 64, 32, 32, 16, 8, 16, 3);
            BENCHMARK(64, 64, 32, 64, 32, 32, 16, 8, 16, 4);

            BENCHMARK(64, 32, 128, 64, 32, 128, 16, 8, 32, 2);
            /* BENCHMARK(64, 32, 128, 64, 32, 128, 16, 8, 32, 3);
            BENCHMARK(64, 32, 128, 64, 32, 128, 16, 8, 32, 4); */

            BENCHMARK(64, 32, 16, 32, 32, 16, 16, 8, 16, 2);

            BENCHMARK(64, 16, 16, 64, 16, 16, 16, 8, 16, 2);
            break;

            case 128:
            BENCHMARK(128, 64, 32, 64, 32, 32, 16, 8, 32, 2);
            BENCHMARK(128, 64, 32, 64, 32, 32, 16, 8, 32, 3);
            BENCHMARK(128, 64, 32, 64, 32, 32, 16, 8, 32, 4);
            BENCHMARK(128, 64, 32, 64, 32, 32, 16, 8, 32, 5);
            BENCHMARK(128, 64, 32, 64, 32, 32, 16, 8, 32, 6);

            BENCHMARK(128, 64, 64, 64, 32, 64, 16, 8, 32, 2);
            BENCHMARK(128, 64, 64, 64, 32, 64, 16, 8, 32, 3);
            BENCHMARK(128, 64, 64, 64, 32, 64, 16, 8, 32, 4);

            BENCHMARK(128, 64, 32, 128, 32, 32, 16, 8, 32, 2);
            BENCHMARK(128, 64, 32, 128, 32, 32, 16, 8, 32, 3);
            BENCHMARK(128, 64, 32, 128, 32, 32, 16, 8, 32, 4);

            BENCHMARK(128, 128, 32, 128, 128, 32, 16, 8, 32, 4);
            BENCHMARK(128, 128, 32, 64, 64, 32, 16, 8, 32, 2);
            BENCHMARK(128, 128, 32, 64, 64, 32, 16, 8, 32, 4);
            BENCHMARK(128, 128, 32, 64, 32, 32, 16, 8, 32, 2);
            BENCHMARK(128, 128, 32, 64, 32, 32, 16, 8, 32, 4);

            //BENCHMARK(128, 32, 128, 128, 32, 128, 16, 8, 32, 4);

            BENCHMARK(128, 64, 16, 128, 32, 16, 16, 8, 16, 2);
            BENCHMARK(128, 64, 16, 128, 32, 16, 16, 8, 16, 4);

            BENCHMARK(128, 16, 16, 64, 16, 16, 16, 8, 16, 2);

            BENCHMARK(128, 16, 16, 128, 16, 16, 16, 8, 16, 2);

            BENCHMARK(128, 32, 32, 128, 32, 32, 16, 8, 32, 2);

            BENCHMARK(128, 128, 32, 128, 64, 32, 16, 8, 32, 2);

            BENCHMARK(128, 128, 32, 128, 128, 32, 16, 8, 32, 2);

	    BENCHMARK(64, 64, 32, 32, 32, 32, 16, 8, 32, 2);
	    BENCHMARK(64, 64, 32, 32, 32, 32, 16, 8, 32, 3);
	    BENCHMARK(64, 64, 32, 32, 32, 32, 16, 8, 32, 4);

	    BENCHMARK(128, 64, 32, 32, 64, 32, 16, 8, 32, 2);
	    BENCHMARK(128, 64, 32, 32, 64, 32, 16, 8, 32, 3);
	    BENCHMARK(128, 64, 32, 32, 64, 32, 16, 8, 32, 4);

	    BENCHMARK(128, 32, 64, 64, 32, 64, 16, 8, 32, 2);
	    BENCHMARK(128, 32, 64, 64, 32, 64, 16, 8, 32, 3);
	    BENCHMARK(128, 32, 64, 64, 32, 64, 16, 8, 32, 4);

	    BENCHMARK(128, 64, 64, 64, 16, 64, 16, 8, 32, 2);
	    BENCHMARK(128, 64, 64, 64, 16, 64, 16, 8, 32, 3);
	    BENCHMARK(128, 64, 64, 64, 16, 64, 16, 8, 32, 4);

	    BENCHMARK(128, 64, 32, 64, 16, 32, 16, 8, 32, 2);
	    BENCHMARK(128, 64, 32, 64, 16, 32, 16, 8, 32, 3);
	    BENCHMARK(128, 64, 32, 64, 16, 32, 16, 8, 32, 4);

	    BENCHMARK(128,32, 32, 32, 32, 32, 16, 8, 32, 2);
	    BENCHMARK(128,32, 32, 32, 32, 32, 16, 8, 32, 3);
	    //BENCHMARK(128,64, 32, 16, 64, 32, 16, 8, 32, 2);
	    //BENCHMARK(128,64, 32, 16, 64, 32, 16, 8, 32, 3);
	    //BENCHMARK(128,64, 32, 16, 64, 32, 16, 8, 32, 4);
	    BENCHMARK(128,64, 32, 32, 32, 32, 16, 8, 32, 2);
	    BENCHMARK(128,64, 32, 32, 32, 32, 16, 8, 32, 3);
	    BENCHMARK(128,64, 32, 32, 32, 32, 16, 8, 32, 4);
	    BENCHMARK(128,128,32, 32, 64, 32, 16, 8, 32, 2);
	    BENCHMARK(128,128,32, 32, 64, 32, 16, 8, 32, 4);
	    BENCHMARK(128,128,32, 64, 32, 32, 16, 8, 32, 2);
	    BENCHMARK(128,128,32, 64, 32, 32, 16, 8, 32, 4);
            break;

            case 256:
            BENCHMARK(256, 64, 16, 64, 32, 16, 16, 8, 16, 2);
            BENCHMARK(256, 32, 16, 64, 32, 16, 16, 8, 16, 2);
            BENCHMARK(256, 32, 32, 64, 32, 32, 16, 8, 32, 2);
            BENCHMARK(256, 32, 32, 64, 32, 32, 16, 8, 16, 2);
            BENCHMARK(256, 64, 16, 64, 64, 16, 16, 8, 16, 2);
            break;
        }

    } else {
        std::cout << "pattern_code: " << pattern_code << std::endl;
        std::cerr << "only pattern:block-2in4 is implemented.\n";
        exit(EXIT_FAILURE);
    }
}
