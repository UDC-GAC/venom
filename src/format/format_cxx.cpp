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

#include "format_cxx.hpp"

template<class T>
Format_cxx<T>::Format_cxx(int nrow, int ncol, int nnz, float density, unsigned seed) :Format<T>(nrow, ncol, nnz, density, seed){};

template<class T>
Format_cxx<T>::~Format_cxx(){
    if (device_synced) {
        __remove_device_ptrs();
    }
};

template<class T>
void Format_cxx<T>::init(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_, bool row_permute_, int brow_, int bcol_, int mbrow, int bm){

    assert(nrow_ % brow_ == 0);
    assert(ncol_ % bcol_ == 0);

    this->A_num_rows = nrow_;
    this->A_num_cols = ncol_;
    this->A_nnz = nnz_;
    this->density = density_;
    this->seed = seed_;
    this->brow = brow_;
    this->bcol = bcol_;
    this->row_permute = row_permute_;
    this->A_size = nrow_*ncol_;
    this->alignment=8;

    Format_cxx<T>::random_init(nrow_, ncol_, brow_, bcol_, density_, this->A_nnz, this->hA_Offsets, this->hA_columns, seed_);
    this->density = (float)this->A_nnz / (nrow_/brow_) / (ncol_/bcol_);

    if (row_permute_) {
        std::vector<int> permutation(nrow_);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(),
                    std::default_random_engine(seed_));
        this->row_permute_ids = permutation;
    }
    else this->row_permute_ids.resize(0);

    this->hA_values.resize(this->A_nnz * brow_ * bcol_);
    for (auto it = this->hA_values.begin(); it != this->hA_values.end(); it++) {
        *it = static_cast<T>((float)(1.0f));
	    //*it = static_cast<T>((float)(std::rand() % 9 - 4));
        //*it = static_cast<T>((float)(std::rand() % 8 + 1));
    }

    this->initialized = true;
    if (this->device_synced) {
        this->__remove_device_ptrs();
        this->device_synced = false;
    }
}

template<typename T>
void Format_cxx<T>::reformat_and_cpy_to_device() {

    assert(this->initialized && "must call init() before copy to device.\n");

    int size_of_meta = 4*(
        (ROUND_UP(this->hA_Offsets.size(), this->alignment))
        + (ROUND_UP(this->hA_columns.size(), this->alignment))
        + (ROUND_UP(this->row_permute_ids.size(), this->alignment)));
    this->meta_h = std::vector<char>(size_of_meta);
    memset(this->meta_h.data(), 0x0, size_of_meta*sizeof(char));

    int offset = 0;
    memcpy(this->meta_h.data(), this->hA_Offsets.data(), sizeof(int)*this->hA_Offsets.size());
    offset += this->hA_Offsets.size();
    while (offset % this->alignment != 0) offset++;

    std::vector<int> ind_tmp = this->hA_columns;
    for (auto it = ind_tmp.begin(); it != ind_tmp.end(); it++)
    {    *it *= bcol; }

    memcpy(this->meta_h.data() + offset * (sizeof(int)/sizeof(char)),
            ind_tmp.data(), sizeof(int)*ind_tmp.size());
    offset += ind_tmp.size();
    while (offset % this->alignment != 0) offset++;

    memcpy(this->meta_h.data() + offset * (sizeof(int)/sizeof(char)),
            this->row_permute_ids.data(), sizeof(int)*this->row_permute_ids.size());

    this->data_h = this->hA_values;

    void *meta_d; T* data_d;
    size_t meta_size = this->meta_h.size() *sizeof(char);
    size_t data_size = this->data_h.size() *sizeof(T);

    CUDA_CHECK( cudaMalloc(&meta_d, meta_size));
    CUDA_CHECK( cudaMalloc(&data_d, data_size));

    CUDA_CHECK( cudaMemcpy(meta_d, this->meta_h.data(),
                    meta_size, cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(data_d, this->data_h.data(),
                    data_size, cudaMemcpyHostToDevice));

    this->device_ptrs._meta = meta_d;
    this->device_ptrs._data = data_d;

    offset = 0;
    this->device_ptrs.csb_indptr = (int*)meta_d;

    offset += this->hA_Offsets.size();
    while (offset % this->alignment != 0) offset++;
    this->device_ptrs.csb_indices = (int*)meta_d + offset;

    if (this->row_permute_ids.size() > 0) {
        offset += this->hA_columns.size();
        while (offset % this->alignment != 0) offset++;
        this->device_ptrs.row_permute_ids = (int*)meta_d + offset;
    }
    else this->device_ptrs.row_permute_ids= nullptr;

    this->device_ptrs.csb_values = data_d;

    device_synced = true;
}


template<typename T>
void Format_cxx<T>::random_init(int nrow, int ncol, int brow, int bcol, float block_density, int  &nnzb, int &padded_nnzb, std::vector<int> &indptr, std::vector<int> &indices, int block_divisor, int block_padding, unsigned seed)
{
    int X = nrow / brow;
    int Y = ncol / bcol;
    int nnzb_group = (int)(std::round(block_density * X * Y / block_divisor));
    nnzb = nnzb_group * block_divisor;

    std::vector<int> idx_group(X * (Y/block_divisor));
    std::iota(idx_group.begin(), idx_group.end(), 0);

    std::shuffle(idx_group.begin(), idx_group.end(), std::default_random_engine(seed));
    std::vector<int> row_nnz_count(X, 0);
    for (auto it = idx_group.begin(); it != idx_group.begin() + nnzb_group; it++) {
        row_nnz_count[(*it) % X] += block_divisor;
    }

    std::vector<std::vector<int>> nz_list(X, std::vector<int>());
    for (int i = 0; i < X; i++) {
        int row_nnz = row_nnz_count[i];
        std::vector<int> cid(Y);
        std::iota(cid.begin(), cid.end(), 0);
        std::shuffle(cid.begin(), cid.end(), std::default_random_engine(seed +i+1));
        nz_list[i] = std::vector<int>(cid.begin(), cid.begin() + row_nnz);
    }
    int offset = 0, padded_offset = 0;
    indptr.clear();
    indices.clear();
    for (int i = 0; i < X; i++) {
        indptr.push_back(padded_offset);
        std::sort(nz_list[i].begin(), nz_list[i].end());
        std::copy(nz_list[i].begin(), nz_list[i].end(), std::back_inserter(indices));
        offset += nz_list[i].size();
        padded_offset += nz_list[i].size();
        while (padded_offset % block_padding != 0) {
            indices.push_back(*(nz_list[i].end() -1));
            padded_offset++;
        }
    }
    assert (offset == nnzb);
    indptr.push_back(padded_offset);
    padded_nnzb = padded_offset;
}

template<typename T>
void Format_cxx<T>::random_init(int nrow, int ncol, int brow, int bcol, float block_density, int &nnzb, std::vector<int> &indptr, std::vector<int> &indices, unsigned seed)
{
    int nnzb_tmp, padded_nnzb_tmp;
    random_init(nrow, ncol, brow, bcol, block_density, nnzb_tmp, padded_nnzb_tmp, indptr, indices, 1, 1, seed);
    assert(nnzb_tmp==padded_nnzb_tmp);
    nnzb = nnzb_tmp;
}

template<class T>
std::vector<int>& Format_cxx<T>::get_hA_Offsets(){
    return this->hA_Offsets;
}

template<class T>
std::vector<int>& Format_cxx<T>::get_hA_col(){
    return this->hA_columns;
}

template<class T>
std::vector<T>& Format_cxx<T>::get_hA_val(){
    return this->hA_values;
}

template<class T>
typename Format_cxx<T>::DevicePtrs& Format_cxx<T>::get_device_ptrs(){
    return this->device_ptrs;
}

template<class T>
void Format_cxx<T>::__remove_device_ptrs() {
    if (device_ptrs._meta)
        CUDA_CHECK( cudaFree(device_ptrs._meta));
    if (device_ptrs._data)
        CUDA_CHECK( cudaFree(device_ptrs._data));
}

template class Format_cxx<float>;
template class Format_cxx<__half>;
template class Format_cxx<int8_t>;
