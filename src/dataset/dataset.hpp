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

#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include "../format/format.hpp"
#include "../format/format_cxx.hpp"
#include "../format/format_csr.hpp"
#include "../format/format_cvs.hpp"
#include "../util/matrix_utils.h"

using namespace std;

template <class T, class T2>
class Dataset {
    public:
        Dataset(Format<T> &format);
        Dataset(int m, int k, int n, float density, Format<T> &format, int seed, int block_sz=1);
        Dataset(int m, int k, int n, float density, Format<T> &format, int seed,int brow_, int bcol_);
        Dataset(int a_m, int a_k, int b_n, float density, Format<T> &format, int seed, int n_, int m_, int brow, int mbrow, int bm_);
        ~Dataset();
        // common methods
        Format<T>* get_format();
        std::vector<T>& get_B();
        std::vector<T2>& get_C();
        Matrix<T>& get_Bmat();
        Matrix<T2>& get_Cmat();
        int get_A_size();
        int get_B_size();
        int get_C_size();
        int get_A_num_rows();
        int get_A_num_cols();
        int get_B_num_rows();
        int get_B_num_cols();

    protected:
        // attributes
        vector<int> A_details;
        int A_num_rows, A_num_cols;
        int B_num_rows, B_num_cols;
        int A_size, B_size, C_size;
        std::vector<T> hB;
        std::vector<T2> hC;
        Format<T>& format;

        RandomMatrix<T> B;
        //OnesMatrix<T> B;
        ZerosMatrix<T2> C;

        // methods
        void set_B(T value);
        void set_C(T2 value);
};

#endif
