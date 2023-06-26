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

#ifndef FORMAT_SR_NM_H
#define FORMAT_SR_NM_H

#include "format.hpp"

template<class T>
class Format_sr_nm: public Format<T> {
    struct DeviceRef {
            void *_meta = nullptr;
            T    *_data = nullptr;
            T    *opdA;
            uint *metadata;
            uint *csb_indices;
    };
    public:
        Format_sr_nm(){};
        Format_sr_nm(int nrow, int ncol, int n, int m, int nnz, float density, unsigned seed);
        ~Format_sr_nm();
        void transform_and_sync_device();
        void init(int nrow_, int ncol_, int n_, float density_, unsigned seed_, bool row_permute_, int m_=32, int brow_=1, int mbrow_=1, int bm_=1);
        void init_from_dense(int nrow_, int ncol_, float density_, unsigned seed_, int n_, int m_, int brow_, int mbrow_, int bm_, std::vector<T>& dense, std::vector<uint>& columns);
        void init_from_sparse(int nrow_, int ncol_, float density_, unsigned seed_, int n_, int m_, int brow_, int mbrow_, int bm_, std::vector<T>& dense, std::vector<uint>& columns_, std::vector<uint>& metadata_);
        std::vector<unsigned int>& get_hA_metadata();
        std::vector<unsigned int>& get_hA_indices();
        std::vector<T>& get_hA_val();
        int get_n();
        int get_m();
        int get_bm();
        int get_vec_length();
        int get_meta_vec_length();
        void to_String();
        std::vector<T>& to_dense() override;
        void to_sparse_sr_nm(std::vector<T>& dense);
        DeviceRef& get_device_ref();
    protected:
        void __clear_device_ref();
        void random_mask(int nrow, int ncol, int ncol_pad, int n, int m, int brow, float density, std::vector<unsigned int> &metadata, std::vector<unsigned int> &indices, unsigned seed, int bm_);

        std::string config_str;
        const int alignment = 8;

        int A_num_cols_sp;
        int A_num_cols_sp_pad_nm;
        int A_num_cols_sp_pad;
        int n, m, brow, mbrow, bm;

        // !IMPORTANT! value fixed because of NVIDIA architecture (2:4)
        int m_fixed, mrow_m, nelems, nelems_col;
        int bits_elem_meta, bits_elem_cols;
        int brow_fixed;

        std::vector<char> meta_h;
        std::vector<T>    data_h;

        bool initialized = false;
        std::vector<uint> hA_metadata;
        std::vector<uint> hA_columns;
        std::vector<T> hA_values;

        bool device_synced = false;
        DeviceRef device_ref;
};

#endif