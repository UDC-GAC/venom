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

#include <iostream>
#include <cfloat>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <cusparseLt.h>

#include "./format/format_cxx.hpp"
#include "./format/format_csr.hpp"
#include "./format/format_cvs.hpp"
#include "./format/format_sr_nm.hpp"

#include "./dataset/dataset.hpp"

#include "./spmm/sputnik_spmm.hpp"
#include "./spmm/clasp_spmm.hpp"
#include "./spmm/spatha_spmm.hpp"
#include "./spmm/cusparseLt_spmm.hpp"

#include "./gemm/cublas_gemm.hpp"
#include "./gemm/cublasLt_gemm.hpp"

#include "./util/argparse.h"

#define TIMES 100

using namespace std;

template<typename T, typename T2>
int check_results(std::vector<T2> cusparse_result, std::vector<T2> cuBLAS_result, Dataset<T,T2>& d, int num_batches){
    int errors = 0;
    for(int i=0; i<num_batches; i++){
        for (int j=0; j < d.get_A_num_rows() * d.get_B_num_cols() ; j++){
            float c_value  = static_cast<float>(cusparse_result[i*(d.get_C_size()+128)+ j]);
            float c_result = static_cast<float>(cuBLAS_result  [i*d.get_C_size()+ j]);
            if (abs(c_value - c_result) > 1.0){
            //if(c_value / c_result > 1.0001 || c_result / c_value > 1.0001 || abs(c_value - c_result) > 1e-5){
                std::cout << "(" << j/d.get_B_num_cols()  << "," << j%d.get_B_num_cols()  << "): " << c_value << " vs " << c_result << std::endl;
                errors ++;
                break;
            }
        }
    }

    if (errors > 0){
        return 1;
    } else {
        return 0;
    }

    return errors;
}

template<typename T, typename T2>
Dataset<T,T2>* create_dataset(int m, int n, int k, float density, Format<T> &fm, int block_sz, int seed, int meta_block_sz=0, int nn_row=0, int mm_row=0, int bm_=1){
    Dataset<T,T2> *dset;

    if(meta_block_sz==0)
        dset = new Dataset<T,T2>(m, k, n, density, fm, seed, block_sz);
    else
        dset = new Dataset<T,T2>(m, k, n, density, fm, seed, nn_row, mm_row, block_sz, meta_block_sz, bm_);

    return dset;
}

template<typename T>
Format<T>* create_sparse_format(int pattern_code){
    Format<T> *fm;
    switch (pattern_code)
    {
    case 0:
        fm = new Format_csr<T>();
        break;
    case 1:
        fm = new Format_cvs<T>();
        break;
    case 2:
        fm = new Format_sr_nm<T>();
        break;
    default:
        break;
    }

    return fm;
}

template<typename T, typename T2, typename T3>
Gemm<T,T2,T3>* create_gemm(int gemm_code, std::vector<T>& A_dense, Dataset<T,T2> &d, cudaDataType_t acc_type, cublasComputeType_t C_gemm){

    Gemm<T,T2,T3>* gemm;
    std::vector<T> dst(d.get_A_size());

    switch (gemm_code)
    {
    case 0:
        gemm = new Cublas_gemm<T,T2,T3>(A_dense, d, acc_type);
        break;
    case 1:
        gemm = new CublasLt_gemm<T,T2,T3>(A_dense, d, 1, acc_type, C_gemm);
        break;

    default:
        break;
    }

    return gemm;
}

template<typename T, typename T2>
Spmm_CNN<T,T2>* create_spmm(int spmm_code, Dataset<T,T2> &d, cudaDataType_t S, cusparseComputeType C_spmm){
    Spmm_CNN<T,T2>* spmm;

    switch (spmm_code)
    {
    case 0:
        spmm = new Clasp_Spmm<T,T2>(d, S, C_spmm);
        break;
    case 1:
        spmm = new Sputnik_Spmm<T,T2>(d, S, C_spmm);
        break;
    case 2:
        spmm = new Spatha<T,T2>(d, S, C_spmm);
        break;
    case 3:
        spmm = new CusparseLt_Spmm<T,T2>(d, S, C_spmm);
        break;

    default:
        break;
    }

    return spmm;
}

template<typename T, typename T2, typename T3>
void launch_kernels(int pattern_code, int gemm_code, int spmm_code, int check, int m, int n, int k, float density, int block_sz, int meta_block_sz, int bm, int bn, int bk, int wm, int wn, int wk, int mm, int mn, int mk, int nstage, int nn_row, int mm_row, int seed, cudaDataType_t acc_type, cublasComputeType_t C_gemm, cusparseComputeType C_spmm){

    Format<T> *fm = create_sparse_format<T>(pattern_code);
    Dataset<T,T2> *dt = create_dataset<T, T2>(m, n, k, density, *fm, block_sz, seed, meta_block_sz, nn_row, mm_row, bm);

    Spmm_CNN<T,T2> *spmm = create_spmm(spmm_code, *dt, acc_type, C_spmm);

    Gemm<T,T2,T3> *gemm;
    if(spmm_code==3){
        gemm = create_gemm<T,T2,T3>(gemm_code, spmm->get_sparse(), *dt, acc_type, C_gemm);
    } else {
        gemm = create_gemm<T,T2,T3>(gemm_code, fm->to_dense(), *dt, acc_type, C_gemm);
    }

    cudaProfilerStart();
    float spmm_time = spmm->spmm(TIMES, bm, bn, bk, wm, wn, wk, mm, mn, mk, nstage);
    float gemm_time = gemm->sgemm(TIMES);
    cudaProfilerStop();

    int error=0;
    if(check)
        error = check_results(gemm->get_C(), spmm->get_result(), *dt, 1);

    cout << spmm_code << "," << "sm_86" << "," << m << "," << k << "," << n << "," << meta_block_sz << "," << block_sz << "," << nn_row << "," << mm_row << "," << density << "," << bm << "," << bn << "," << bk << "," << wm << "," << wn << "," << wk << "," << mm << "," << mn << "," << mk << "," << nstage << "," << spmm_time << "," << gemm_time << "," << gemm_time/spmm_time << "," << error << endl;

    delete gemm;
    delete spmm;
    delete fm;
    //delete dt;
}

int main(int argc, const char **argv) {
    int m, n, k;
    int pattern_code, block_sz, meta_block_sz;
    float density;
    unsigned seed;
    int spmm_code, gemm_code, precision_code, acc_code;
    bool check;
    int bm, bn, bk;
    int wm, wn, wk;
    int mm, mn, mk;
    int nstage;
    int nn_row, mm_row;
    bool load_pattern;
    std::string path_to_dset;

    parseArgs(argc, argv, m, n, k, density, block_sz, meta_block_sz, spmm_code, gemm_code, acc_code, pattern_code, precision_code, seed, check, bm, bn, bk, wm, wn, wk, mm, mn, mk, nstage, nn_row, mm_row, load_pattern, path_to_dset, /* verbose */false);

    switch (precision_code)
    {
        case 0:
            launch_kernels<float, float, float>(pattern_code, gemm_code, spmm_code, check, m, n, k, density, block_sz, meta_block_sz, bm, bn, bk, wm, wn, wk, mm, mn, mk, nstage, nn_row, mm_row, seed, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUSPARSE_COMPUTE_TF32_FAST);
            break;
        case 1:
            launch_kernels<half, half, float>(pattern_code, gemm_code, spmm_code, check, m, n, k, density, block_sz, meta_block_sz, bm, bn, bk, wm, wn, wk, mm, mn, mk, nstage, nn_row, mm_row, seed, CUDA_R_32F, CUBLAS_COMPUTE_16F, CUSPARSE_COMPUTE_16F);
            break;
        case 2:
            launch_kernels<half, half, half>(pattern_code, gemm_code, spmm_code, check, m, n, k, density, block_sz, meta_block_sz, bm, bn, bk, wm, wn, wk, mm, mn, mk, nstage, nn_row, mm_row, seed, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUSPARSE_COMPUTE_16F);
            break;
        case 3:
            launch_kernels<int8_t, int8_t, int8_t>(pattern_code, gemm_code, spmm_code, check, m, n, k, density, block_sz, meta_block_sz, bm, bn, bk, wm, wn, wk, mm, mn, mk, nstage, nn_row, mm_row, seed, CUDA_R_8I, CUBLAS_COMPUTE_32I, CUSPARSE_COMPUTE_32I);
            break;
    }

}