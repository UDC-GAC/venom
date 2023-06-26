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

#include "gemm.hpp"

template<class T, class T2, class T3>
Gemm<T,T2,T3>::Gemm(std::vector<T>& A_dense, Dataset<T,T2> &d)
    :dset(d)
{
    A = A_dense;
    B = d.get_B();
    //C = d.get_C();

    A_size = d.get_A_size();
    B_size = d.get_B_size();
    C_size = d.get_C_size();
    A_num_rows = d.get_A_num_rows();
    A_num_cols = d.get_A_num_cols();
    B_num_rows = d.get_A_num_cols();
    B_num_cols = d.get_B_num_cols();

    for (int i = 0; i < C_size; i++) {
        C.push_back(static_cast<T>(0.0f));
    }

    alpha = 1.0f;
    beta  = 0.0f;

    cudaMalloc((void**) &dA, A_size * sizeof(T));
    cudaMalloc((void**) &dB, B_size * sizeof(T));
    cudaMalloc((void**) &dC, C_size * sizeof(T));

    cudaMemcpy(dA, &A[0], A_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &B[0], B_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dC, &C[0], C_size * sizeof(T),
                           cudaMemcpyHostToDevice);
}

template<class T, class T2, class T3>
Gemm<T,T2,T3>::Gemm(std::vector<T>& A_dense, Dataset<T,T2> &d, int batch_size, cudaDataType_t Tp, cublasComputeType_t S)
    :dset(d)
{
    A = A_dense;
    B = d.get_B();
    C = d.get_C();

    A_size = d.get_A_size();
    B_size = d.get_B_size();
    C_size = d.get_C_size();
    A_num_rows = d.get_A_num_rows();
    A_num_cols = d.get_A_num_cols();
    B_num_rows = d.get_A_num_cols();
    B_num_cols = d.get_B_num_cols();

    alpha = 1.0f;
    beta  = 0.0f;

    cudaMalloc((void**) &dA, A_size * sizeof(T));
    cudaMalloc((void**) &dB, B_size * sizeof(T));
    cudaMalloc((void**) &dC, C_size * sizeof(T2));

    cudaMemcpy(dA, &A[0], A_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &B[0], B_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dC, &C[0], C_size * sizeof(T2),
                           cudaMemcpyHostToDevice);
}

template<class T, class T2, class T3>
Gemm<T,T2,T3>::~Gemm(){
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

template<class T, class T2, class T3>
std::vector<T2>& Gemm<T,T2,T3>::get_C(){
    return C;
}

template class Gemm<float, float, float>;
template class Gemm<__half, __half, float>;
template class Gemm<__half, __half, __half>;
template class Gemm<int8_t, int8_t, int8_t>;