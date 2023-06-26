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

#include "cublas_gemm.hpp"

template<class T, class T2, class T3>
Cublas_gemm<T,T2,T3>::Cublas_gemm(std::vector<T>& A_dense, Dataset<T,T2> &d, cudaDataType_t acc_type)
    :Gemm<T,T2,T3>(A_dense, d)
{
    cublasCreate(&handle);
    this->acc_type=acc_type;
}

template<class T, class T2, class T3>
Cublas_gemm<T,T2,T3>::~Cublas_gemm(){
    cublasDestroy(handle);
}

template<>
inline float Cublas_gemm<float,float,float>::sgemm(int times, int num_batches){
    float time;

    ///////////////////////
    time = cuTime(times, cublasSgemm, handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                this->B_num_cols, this->A_num_rows, this->B_num_rows,
                &(this->alpha),
                this->dB, this->B_num_cols,
                this->dA, this->B_num_rows,
                &(this->beta),
                this->dC, this->B_num_cols);

    float *hC = &(this->get_C()[0]);
    cudaMemcpy(hC, this->dC, this->dset.get_C_size() * sizeof(float),
                           cudaMemcpyDeviceToHost);

    return time;
}

template<>
inline float Cublas_gemm<half, half, float>::sgemm(int times, int num_batches){
    float time;
    // Performs warmup operation
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    int warmup=(times>0)?(10):(0);

    cudaEvent_t start, stop;
    ///////////////////////

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i=0; i<warmup+times; ++i){
        if (i == warmup)
            cudaEventRecord(start, 0);

        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            this->B_num_cols, this->A_num_rows, this->B_num_rows,
            &this->alpha,
            this->dB, CUDA_R_16F, this->B_num_cols,
            this->dA, CUDA_R_16F, this->B_num_rows,
            &this->beta,
            this->dC, CUDA_R_16F, this->B_num_cols,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time = time / (float)times;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    half *hC = &(this->get_C()[0]);
    cudaMemcpy(hC, this->dC, this->dset.get_C_size() * sizeof(half),
                           cudaMemcpyDeviceToHost);

    return time;
}

template<>
inline float Cublas_gemm<half, half, half>::sgemm(int times, int num_batches){
    float time;
    // Performs warmup operation
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    int warmup=(times>0)?(10):(0);

    cudaEvent_t start, stop;
    ///////////////////////

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i=0; i<warmup+times; ++i){
        if (i == warmup)
            cudaEventRecord(start, 0);

        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            this->B_num_cols, this->A_num_rows, this->B_num_rows,
            &this->alpha,
            this->dB, CUDA_R_16F, this->B_num_cols,
            this->dA, CUDA_R_16F, this->B_num_rows,
            &this->beta,
            this->dC, CUDA_R_16F, this->B_num_cols,
            CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time = time / (float)times;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    half *hC = &(this->get_C()[0]);
    cudaMemcpy(hC, this->dC, this->dset.get_C_size() * sizeof(half),
                           cudaMemcpyDeviceToHost);

    return time;
}

template<>
inline float Cublas_gemm<int8_t, int8_t, int8_t>::sgemm(int times, int num_batches){
    std::cerr << "Operation not supported yet\n";
    exit(EXIT_FAILURE);
}

template class Cublas_gemm<float, float, float>;
template<> float Cublas_gemm<float, float, float>::sgemm(int times, int num_batches);
template class Cublas_gemm<half, half, float>;
template<> float Cublas_gemm<half, half, float>::sgemm(int times, int num_batches);
template class Cublas_gemm<half, half, half>;
template<> float Cublas_gemm<half, half, half>::sgemm(int times, int num_batches);
template class Cublas_gemm<int8_t, int8_t, int8_t>;
template<> float Cublas_gemm<int8_t, int8_t, int8_t>::sgemm(int times, int num_batches);
