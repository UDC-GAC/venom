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

#ifndef GEMM_H
#define GEMM_H

#include <string>
#include <iostream>
#include "../dataset/dataset.hpp"
#include "../cuda_utils.h"
#include <cublasLt.h>

using namespace std;

template<class T, class T2, class T3>
class Gemm {
    public:
        Gemm(std::vector<T>& A_dense, Dataset<T,T2> &d);
        Gemm(std::vector<T>& A_dense, Dataset<T,T2> &d, int batch_size, cudaDataType_t Tp, cublasComputeType_t S);
        ~Gemm();
        virtual float sgemm(int times, int num_batches=1) = 0;
        std::vector<T2>& get_C();

    protected:
        int A_size, B_size, C_size;
        int A_num_rows, A_num_cols;
        int B_num_rows, B_num_cols;
        T3 alpha;
        T3 beta;
        Dataset<T,T2>& dset = nullptr;
        std::vector<T> A;
        std::vector<T> B;
        std::vector<T2> C;
        T *dA;
        T *dB;
        T2 *dC;
};

#endif