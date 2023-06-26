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

#ifndef CLASP_SPMM_H
#define CLASP_SPMM_H

#include "spmm.hpp"

#include "../../include/CLASP/include/wmma_spmm.cuh"
#include "../../include/sputnik/sputnik/spmm/cuda_spmm.h"
#include "../format/format_cvs.hpp"

using namespace std;

template<class T, class T2>
class Clasp_Spmm: public Spmm_CNN<T,T2> {
    public:
        ~Clasp_Spmm();
        Clasp_Spmm(Dataset<T,T2> &d, cudaDataType_t S, cusparseComputeType C_spmm);
        std::vector<T>& get_hB();
        std::vector<T2>& get_hC();
        std::vector<T2>& get_result();
        std::vector<T>& get_sparse();
        float spmm(int times, int bm_=0, int bn_=0, int bk_=0, int wm_=0, int wn_=0, int wk_=0, int mm_=0, int mn_=0, int mk_=0, int nstage_=0);

    private:
        void SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices);
        void IdentityRowSwizzle(int rows, int *row_indices);
        int* dA_RowIndex;
        std::vector<T2> result;
};

#endif