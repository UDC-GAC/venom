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

#ifndef CUBLASLT_GEMM_H
#define CUBLASLT_GEMM_H

#include "gemm.hpp"
#include <cublasLt.h>

using namespace std;

template<class T, class T2, class T3>
class CublasLt_gemm: public Gemm<T,T2,T3>{
    public:
        CublasLt_gemm(std::vector<T>& A_dense, Dataset<T,T2> &d, int batch_size, cudaDataType_t Tp, cublasComputeType_t S);
        ~CublasLt_gemm();

        float sgemm(int times, int num_batches=1);

    private:
        int batch_size;
        cublasLtHandle_t ltHandle;
        cudaStream_t stream;
        void *workspace;
        size_t workspaceSize;
        std::vector<T> A_lt, B_lt, biasHost;
        std::vector<T2> C_lt;
        T *biasDev, *Adev, *Bdev;
        T2 *Cdev;
        int64_t stridea, strideb, stridec;
        float alpha, beta; //FIXME: int8
        //half alpha, beta; //FIXME: half

        cublasLtMatmulDesc_t operationDesc = NULL;
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
        cublasLtMatmulPreference_t preference = nullptr;
        int                             returnedResults = 0;
        cublasLtMatmulHeuristicResult_t heuristicResult = {};
};

#endif
