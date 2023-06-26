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

#ifndef FORMAT_CXX_H
#define FORMAT_CXX_H

#include "format.hpp"


template<class T>
class Format_cxx: public Format<T> {
    struct DevicePtrs {
        void *_meta = nullptr;
        T    *_data = nullptr;
        int  *row_permute_ids;
        int  *csb_indptr;
        int  *csb_indices;
        T    *csb_values;
    };

    public:
        Format_cxx(){};
        Format_cxx(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_);
        ~Format_cxx();
        void reformat_and_cpy_to_device();
        void init(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_, bool row_permute_, int brow_=1, int bcol_=1, int mbrow=0, int bm=1);
        std::vector<int>& get_hA_Offsets();
        std::vector<int>& get_hA_col();
        std::vector<T>& get_hA_val();
        DevicePtrs& get_device_ptrs();
    protected:
        void __remove_device_ptrs();
        void random_init(int nrow_, int ncol_, int brow_, int bcol_, float density_, int &nnz_, std::vector<int> &indptr_, std::vector<int> &indices_, unsigned seed_);
        void random_init(int nrow_, int ncol_, int brow_, int bcol_, float density_, int  &nnz_, int &padded_nnz_, std::vector<int> &indptr_, std::vector<int> &indices_, int block_divisor_, int block_padding_, unsigned seed_);

        bool initialized = false;
        std::vector<int> hA_Offsets, hA_columns;
        std::vector<T> hA_values;

        std::string config_str;
        //const int alignment = 8;
        int alignment;

        int brow, bcol;

        bool row_permute;
        std::vector<int> row_permute_ids;
        std::vector<char> meta_h;
        std::vector<T>    data_h;

        bool device_synced = false;
        DevicePtrs device_ptrs;
};

#endif