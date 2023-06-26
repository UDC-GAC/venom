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
#include "./cusparseLt_spmm.hpp"
#include "../dataset/dataset.hpp"

#include <cuda_runtime_api.h>
//#include <cusparseLt.h>       // cusparseLt header
#include <cusparse.h>
#include <cstdio>             // printf
#include <cstdlib>            // std::rand

#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "../cuda_utils.h"


template<class T, class T2>
CusparseLt_Spmm<T,T2>::CusparseLt_Spmm(Dataset<T,T2> &d, cudaDataType_t S, cusparseComputeType C_spmm)
    :Spmm_CNN<T,T2>(d){

    Format_csr<T> *f = dynamic_cast<Format_csr<T>*>(d.get_format());

    alpha = 1;
    beta = 0;

    this->hA = f->to_dense(); //hA_v;
    this->hB = d.get_B();
    this->hC = d.get_C();

    num_batches   = 1;//batches;
    int m             = d.get_A_num_rows();
    int n             = d.get_B_num_cols();
    int k             = d.get_B_num_rows();
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    /* auto          type  = CUDA_R_16F;
    auto          compute_type = CUSPARSE_COMPUTE_16F; */
    /* auto          type  = CUDA_R_16F;
    auto          compute_type = CUSPARSE_COMPUTE_16F; */
    auto          type  = CUDA_R_16F;
    auto          compute_type = C_spmm;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    num_A_rows     = (isA_transposed) ? k : m;
    num_A_cols     = (isA_transposed) ? m : k;
    num_B_rows     = (isB_transposed) ? n : k;
    num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    int64_t batch_strideA = A_height * lda + 128; //TODO: 128? As a Buffer?? Or alignment?
    int64_t batch_strideB = B_height * ldb + 128;
    int64_t batch_strideC = C_height * ldc + 128;
    A_size         = num_batches * batch_strideA * sizeof(T);
    B_size         = num_batches * batch_strideB * sizeof(T);
    C_size         = num_batches * batch_strideC * sizeof(T2);

    /*T *dA = this->get_dA();
    T *dB = this->get_dB();
    T *dC = this->get_dC();
    T *dD = this->get_dD();
    T *dA_compressed = this->get_dA_compressed();*/
    for(int i=0; i<128; i++){
        hA.push_back( T(0.0f) );
    }
    for(int i=0; i<128; i++){
        hB.push_back( T(0.0f) );
    }
    for(int i=0; i<128; i++){
        hC.push_back( T2(0.0f) );
    }
    for (int b = 1; b < num_batches; b++) {
        for (int i = 0; i < A_height; i++) {
            for (int j = 0; j < lda; j++)
                //hA[b * batch_strideA + i * lda + j] = random_half_gen();
                hA.push_back( hA[i*lda+j] );
        }
        for(int i=0; i<128; i++){
            hA.push_back( T(0.0f) );
        }
    }
    for (int b = 1; b < num_batches; b++) {
        for (int i = 0; i < B_height; i++) {
            for (int j = 0; j < ldb; j++)
                //hB[b * batch_strideB + i * ldb + j] = random_half_gen();
                hB.push_back( hB[i*ldb+j] );
        }
        for(int i=0; i<128; i++){
            hB.push_back( T(0.0f) );
        }
    }
    for (int b = 1; b < num_batches; b++) {
        for (int i = 0; i < C_height; i++) {
            for (int j = 0; j < ldc; j++)
                //hB[b * batch_strideB + i * ldb + j] = random_half_gen();
                hC.push_back( hC[i*ldc+j] );
        }
        for(int i=0; i<128; i++){
            hC.push_back( T2(0.0f) );
        }
    }

    T *hA_p = &(hA[0]);
    T *hB_p = &(hB[0]);
    T2 *hC_p = &(hC[0]);
    int    *d_valid;

    CUDA_CHECK( cudaMalloc((void**) &dA, A_size) )
    CUDA_CHECK( cudaMalloc((void**) &dB, B_size) )
    CUDA_CHECK( cudaMalloc((void**) &dC, C_size) )
    CUDA_CHECK( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    CUDA_CHECK( cudaMemcpy(dA, hA_p, A_size, cudaMemcpyHostToDevice) )
    CUDA_CHECK( cudaMemcpy(dB, hB_p, B_size, cudaMemcpyHostToDevice) )
    CUDA_CHECK( cudaMemcpy(dC, hC_p, C_size, cudaMemcpyHostToDevice) )

    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cudaStream_t                   stream = nullptr;

    CHECK_CUSPARSELT( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSELT( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSELT( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSELT( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    //--------------------------------------------------------------------------
    // SET NUM BATCHES
    CHECK_CUSPARSELT( cusparseLtMatDescSetAttribute(&handle, &matA,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    CHECK_CUSPARSELT( cusparseLtMatDescSetAttribute(&handle, &matB,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    CHECK_CUSPARSELT( cusparseLtMatDescSetAttribute(&handle, &matC,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    //--------------------------------------------------------------------------
    // SET BATCH STRIDE
    // if batch_strideA = 0, the matrix multiplication performs a broadcast of
    // the matrix A
    CHECK_CUSPARSELT(  cusparseLtMatDescSetAttribute(&handle, &matA,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideA,
                                                sizeof(batch_strideA)) )
    CHECK_CUSPARSELT(  cusparseLtMatDescSetAttribute(&handle, &matB,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideB,
                                                sizeof(batch_strideB)) )
    CHECK_CUSPARSELT(  cusparseLtMatDescSetAttribute(&handle, &matC,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideC,
                                                sizeof(batch_strideC)) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSELT( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    CHECK_CUSPARSELT( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSELT( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSELT( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size) )

    CHECK_CUSPARSELT( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))

    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correcteness
    CHECK_CUSPARSELT( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSELT( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    int is_valid;
    CUDA_CHECK( cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid),
                                cudaMemcpyDeviceToHost, stream) )
    CUDA_CHECK( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        //return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    CHECK_CUSPARSELT( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    //std::cout << compressed_size << std::endl;
    CUDA_CHECK( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSELT( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )

    cudaFree(d_valid);
    //CHECK_CUSPARSELT( cusparseLtMatulDescriptorDestroy(matmul) )

    cudaMemcpy(hA_p, this->dA, A_size, cudaMemcpyDeviceToHost);
}

template<class T, class T2>
CusparseLt_Spmm<T,T2>::~CusparseLt_Spmm(){
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSELT( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSELT( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSELT( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSELT( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSELT( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CUDA_CHECK( cudaFree(dA_compressed) )
    CUDA_CHECK( cudaFree(dA) )
    CUDA_CHECK( cudaFree(dB) )
    CUDA_CHECK( cudaFree(dC) )
}

template<class T, class T2>
std::vector<T>& CusparseLt_Spmm<T,T2>::get_sparse(){
    return hA;
}

template<class T, class T2>
std::vector<T>& CusparseLt_Spmm<T,T2>::get_hB(){
    return hB;
}

template<class T, class T2>
std::vector<T2>& CusparseLt_Spmm<T,T2>::get_result(){
    return hC;
}

template<class T, class T2>
float CusparseLt_Spmm<T,T2>::spmm(int times, int bm_, int bn_, int bk_, int wm_, int wn_, int wk_, int mm_, int mn_, int mk_, int nstage_){
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    float time;

    time = cuTime(times, cusparseLtMatmul, &handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD, d_workspace, streams, num_streams);
    cudaDeviceSynchronize();

    T2 *hC_p = &(hC[0]);
    cudaMemcpy(hC_p, this->dC, C_size, cudaMemcpyDeviceToHost);

    return time;
}

template class CusparseLt_Spmm<int8_t, int8_t>;
template class CusparseLt_Spmm<__half, __half>;
template class CusparseLt_Spmm<float, float>;
