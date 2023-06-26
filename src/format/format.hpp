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

#ifndef FORMAT_H
#define FORMAT_H

#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cassert>
#include <iterator>
#include <sstream>
#include <algorithm>

#include "../util/utils.h"
template<class T>
class Format{
    public:
        Format();
        ~Format();
        Format(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_);
        int get_A_num_rows();
        int get_A_num_cols();
        int get_A_nnz();
        virtual void init(int nrow, int ncol, int nnz, float density, unsigned seed, bool row_permute, int brow, int bcol, int mbrow=1, int bm=1){};
        virtual std::vector<T>& to_dense() = 0;
    protected:
        int A_num_rows, A_num_cols, A_size, A_nnz;
        std::vector<T> hA_dense;
        unsigned seed;
        float density;
};

#endif