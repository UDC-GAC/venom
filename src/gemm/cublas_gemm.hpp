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

#ifndef CUBLAS_GEMM_H
#define CUBLAS_GEMM_H

#include "gemm.hpp"
#include <cublas_v2.h>

using namespace std;

template<class T, class T2, class T3>
class Cublas_gemm: public Gemm<T,T2,T3>{
    public:
        Cublas_gemm(std::vector<T>& A_dense, Dataset<T,T2> &d, cudaDataType_t acc_type);
        ~Cublas_gemm();
        float sgemm(int times, int num_batches=1);

    private:
        cublasHandle_t handle;
        cudaDataType_t acc_type;
};

#endif