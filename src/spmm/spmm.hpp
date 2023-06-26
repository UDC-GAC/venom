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

#ifndef SPMM_H
#define SPMM_H

#include <string>
#include <map>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cusparseLt.h>       // cusparseLt header
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>

#include "../dataset/dataset.hpp"
#include "../format/format.hpp"
#include "../cuda_utils.h"

using namespace std;

template<class T, class T2>
class Spmm_CNN {
    public:
        Spmm_CNN(Dataset<T,T2> &d);
        Spmm_CNN(Dataset<T,T2> &d, cudaDataType_t S, cusparseComputeType C_spmm);
        ~Spmm_CNN();
        T* get_dB();
        T2* get_dC();
        void to_String();
        Dataset<T,T2>& get_dataset();

        virtual float spmm(int times, int bm_=0, int bn_=0, int bk_=0, int wm_=0, int wn_=0, int wk_=0, int mm_=0, int mn_=0, int mk_=0, int nstage_=0) = 0;
        virtual std::vector<T2>& get_result() = 0;
        virtual std::vector<T>& get_sparse() = 0;

    protected:
        Dataset<T,T2>& dset;
        T *dB;
        T2 *dC;
        float gflop_count;
};

#endif
