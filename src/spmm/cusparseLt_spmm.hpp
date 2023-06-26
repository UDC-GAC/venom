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

#ifndef CUSPARSELT_SPMM_H
#define CUSPARSELT_SPMM_H

#include "spmm.hpp"

using namespace std;

template<class T, class T2>
class CusparseLt_Spmm: public Spmm_CNN<T,T2> {
    public:
        // Empty virtual destructor for proper cleanup
        ~CusparseLt_Spmm();
        CusparseLt_Spmm(Dataset<T,T2> &d, cudaDataType_t S, cusparseComputeType C_spmm);

        std::vector<T>& get_sparse();
        std::vector<T>& get_hB();
        std::vector<T2>& get_result();

        float spmm(int times, int bm_=0, int bn_=0, int bk_=0, int wm_=0, int wn_=0, int wk_=0, int mm_=0, int mn_=0, int mk_=0, int nstage_=0);

    private:
        int A_size, B_size, C_size;
        int num_A_rows, num_A_cols, num_B_rows, num_B_cols;
        int num_batches;
        float alpha, beta;

        std::vector<T> hA, hB;
        std::vector<T2> hC;
        T *dA, *dB, *dA_compressed;
        T2 *dC, *dD;

        cusparseLtHandle_t     handle;
        cusparseLtMatDescriptor_t matA, matB, matC;
        cusparseLtMatmulPlan_t plan;
};

#endif