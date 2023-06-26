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
#include "cublasLt_gemm.hpp"
#include "../dataset/dataset.hpp"

#include "../cuda_utils.h"

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

template<class T, class T2, class T3>
CublasLt_gemm<T,T2,T3>::CublasLt_gemm(std::vector<T>& A_dense, Dataset<T,T2> &d, int batch_size, cudaDataType_t Tp, cublasComputeType_t S)
    :Gemm<T,T2,T3>(A_dense, d)
{
    /* A = A_dense;
    B = d.get_B();
    C = d.get_C(); */

    workspaceSize = 1024 * 1024 * 8;
    this->batch_size = batch_size;
    alpha = 1;
    beta  = 0;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

    int m             = d.get_A_num_rows();
    int k             = d.get_A_num_cols();
    int n             = d.get_B_num_cols();
    bool     is_rowmajor    = (rowOrder == CUBLASLT_ORDER_ROW);
    bool     isA_transposed = (transa != CUBLAS_OP_N);
    bool     isB_transposed = (transb != CUBLAS_OP_N);
    auto num_A_rows     = (isA_transposed) ? k : m;
    auto num_A_cols     = (isA_transposed) ? m : k;
    auto num_B_rows     = (isB_transposed) ? n : k;
    auto num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    auto lda = (transa == CUBLAS_OP_N) ? this->A_num_cols : this->A_num_rows;
    //auto     lda            = (!is_rowmajor) ? num_A_cols : num_A_rows;
    auto ldb = (transa == CUBLAS_OP_N) ? this->B_num_cols : this->B_num_rows;
    //auto     ldb            = (!is_rowmajor) ? num_B_cols : num_B_rows;
    auto ldc = (transa == CUBLAS_OP_N) ? this->B_num_cols : this->A_num_rows;
    //auto     ldc            = (!is_rowmajor) ? num_C_cols : num_C_rows;
    stridea = this->A_size;
    strideb = this->B_size;
    stridec = this->C_size;

    //for (int i = 0; i < A_num_rows * batch_size; i++) biasHost.push_back( T(0.0f) );

    for (int b = 0; b < batch_size; b++) {
        //for (int i = 0; i < this->A_num_rows; i++) {
        for (int i = 0; i < num_A_rows; i++) {
            for (int j = 0; j < num_A_cols; j++){
                //hA[b * batch_strideA + i * lda + j] = random_half_gen();
                A_lt.push_back( A_dense[i*num_A_cols+j] );
            }
        }
    }
    std::vector<T> B_cpy = d.get_B();
    for (int b = 0; b < batch_size; b++) {
        //for (int i = 0; i < this->B_num_rows; i++) {
        for (int i = 0; i < num_B_rows; i++) {
            for (int j = 0; j < num_B_cols; j++){
                //hB[b * batch_strideB + i * ldb + j] = random_half_gen();
                B_lt.push_back( B_cpy[i*num_B_cols+j] );
            }
        }
    }
    for (int b = 0; b < batch_size; b++) {
        //for (int i = 0; i <this->A_num_rows; i++) {
        for (int i = 0; i < num_C_rows; i++) {
            for (int j = 0; j < num_C_cols; j++)
                //hB[b * batch_strideB + i * ldb + j] = random_half_gen();
                C_lt.push_back( 0 );
        }
    }

    checkCublasStatus(cublasLtCreate(&ltHandle));
    checkCudaStatus(cudaMalloc(( void ** )&workspace, workspaceSize));
    //checkCudaStatus(cudaStreamCreate(&stream));
    //checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDev), this->A_num_rows * batch_size * sizeof(T)));

    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), this->A_size * batch_size * sizeof(T)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), this->B_size * batch_size * sizeof(T)));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), this->C_size * batch_size * sizeof(T2)));

    /* checkCudaStatus(cudaMemcpyAsync(Adev, &A_lt[0], this->A_size*batch_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(Bdev, &B_lt[0], this->B_size*batch_size * sizeof(T), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(Cdev, &C_lt[0], this->C_size*batch_size * sizeof(T2), cudaMemcpyHostToDevice, stream)); */
    //checkCudaStatus(cudaMemcpyAsync(biasDev, &biasHost[0], A_num_rows*batch_size * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(Adev, &A_lt[0], this->A_size*batch_size * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(Bdev, &B_lt[0], this->B_size*batch_size * sizeof(T), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(Cdev, &C_lt[0], this->C_size*batch_size * sizeof(T2), cudaMemcpyHostToDevice));

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    //checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, S, CUDA_R_16F)); //FIXME: half
    //checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, S, CUDA_R_32I)); //FIXME: int8

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    //checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    //checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    //checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F));

    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // create matrix descriptors, we need to configure batch size and counts in this case
    //checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, Tp, num_A_rows, num_A_cols, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, Tp, transa == CUBLAS_OP_N ? this->A_num_rows : this->A_num_cols, transa == CUBLAS_OP_N ? this->A_num_cols : this->A_num_rows, lda));
    //checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, Tp, num_B_rows , num_B_cols, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, Tp, transb == CUBLAS_OP_N ? this->B_num_rows : this->B_num_cols, transb == CUBLAS_OP_N ? this->B_num_cols : this->B_num_rows, ldb));
    //checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, Tp, num_C_rows, num_C_cols, ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, Tp, this->A_num_rows, this->B_num_cols, ldc));

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
    /* checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea))); */

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
    /* checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb))); */

    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
    /* checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_size, sizeof(batch_size)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec))); */

    // Create preference handle; In general, extra attributes can be
    // used here to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C. However, for simplicity
    // here we assume A,B,C are always well aligned (e.g., directly
    // come from cudaMalloc)
    checkCublasStatus( cublasLtMatmulPreferenceCreate( &preference ) );
    checkCublasStatus( cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof( workspaceSize ) ) );

    // We just need the best available heuristic to try and run matmul.
    // There is no guarantee that this will work. For example, if A is
    // badly aligned, you can request more (e.g. 32) algos and try to
    // run them one by one until something works.
    checkCublasStatus( cublasLtMatmulAlgoGetHeuristic(
            ltHandle,
            operationDesc,
            Adesc,
            Bdesc,
            Cdesc,
            Cdesc,
            preference,
            1,
            &heuristicResult,
            &returnedResults ) );
    //std::cout<<"List of algos :"<<returnedResults<<std::endl;
}

template<class T, class T2, class T3>
CublasLt_gemm<T,T2,T3>::~CublasLt_gemm(){
    checkCublasStatus(cublasLtDestroy(ltHandle));
    checkCudaStatus(cudaFree(Adev));
    checkCudaStatus(cudaFree(Bdev));
    checkCudaStatus(cudaFree(Cdev));
    checkCudaStatus(cudaFree(biasDev));
    checkCudaStatus(cudaFree(workspace));
    //checkCudaStatus(cudaStreamDestroy(stream));

    checkCublasStatus( cublasLtMatmulPreferenceDestroy( preference ) );
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

template<class T, class T2, class T3>
float CublasLt_gemm<T,T2,T3>::sgemm(int times, int num_batches){
    float time=0;
    int warmup=(times>0)?(10):(0);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i=0; i<warmup+times; ++i){
        if (i == warmup)
            cudaEventRecord(start, 0);

        checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     &alpha,
                                     Adev,
                                     Adesc,
                                     Bdev,
                                     Bdesc,
                                     &beta,
                                     Cdev,
                                     Cdesc,
                                     Cdev,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time = time / (float)times;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    //std::cout << "cuBlasLt time: " << time << std::endl;

    T2 *hC = &(this->get_C()[0]);
    checkCudaStatus(cudaMemcpy(hC, Cdev, this->C_size*batch_size * sizeof(T2), cudaMemcpyDeviceToHost));
    //std::cout << "cublasLt stridec: " << stridec << std::endl;
    //checkCudaStatus(cudaMemcpy(hC, Cdev, stridec*batch_size * sizeof(T), cudaMemcpyDeviceToHost));

    return time;
}

template class CublasLt_gemm<int8_t, int8_t, int8_t>;
template class CublasLt_gemm<__half, __half, float>;
template class CublasLt_gemm<__half, __half, __half>;
template class CublasLt_gemm<float, float, float>;
