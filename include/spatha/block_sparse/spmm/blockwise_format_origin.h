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

#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>             // for generating row_indices
#include <algorithm>
#include <random>
#include <chrono>
#include "../common/base.h"        // ROUND_UP, CEIL
#include "../cuda_error.h"         // CUDA_CHECK
#include "../util/random_mask.h"

template<typename T>
struct BlockwiseSpMatrix {
    // shape and pattern configs
    int nrow;
    int ncol;
    int nnzb;
    int brow;
    int bcol;
    float density;
    unsigned seed;
    bool row_permute ;

    // original format on host
    const int alignment = 8;
    std::vector<int> indptr, indices, row_permute_ids;
    std::vector<T> values;

    // transformed format on host
    std::vector<char> meta_h;
    std::vector<T>    data_h;

    // pointers to transformed format on device
    struct DeviceRef {
        void *_meta = nullptr;
        T    *_data = nullptr;
        int  *csb_indptr;
        int  *csb_indices;
        int  *row_permute_ids;
        T    *csb_values;
    } device_ref;

    bool initialized = false;
    bool device_synced = false;
    std::string config_str;

    void __clear_device_ref() {
        if (device_ref._meta)
            CUDA_CHECK( cudaFree(device_ref._meta));
        if (device_ref._data)
            CUDA_CHECK( cudaFree(device_ref._data));
    }
    ~BlockwiseSpMatrix() {
        if (device_synced) {
            __clear_device_ref();
        }
    }

    void init_random(int nrow_, int ncol_, int brow_, int bcol_,
    float expected_density, bool row_permute_, unsigned seed_) ;

    void init_sparse(int nrow_, int ncol_, int brow_, int bcol_,
    int nnz_, bool row_permute_, const std::vector<int>& hA_cvsOffsets, const std::vector<int>& hA_columns, std::vector<T>& hA_values, unsigned seed_);

    void transform_and_sync_device();
};

template<typename T>
void BlockwiseSpMatrix<T>::init_random(int nrow_, int ncol_, int brow_,
int bcol_, float expected_density, bool row_permute_, unsigned seed_)  {

    assert(nrow_ % brow_ == 0);
    assert(ncol_ % bcol_ == 0);

    this->nrow = nrow_;
    this->ncol = ncol_;
    this->brow = brow_;
    this->bcol = bcol_;
    this->seed = seed_;
    this->row_permute = row_permute_;

    // randomly generate a mask
    random_mask(nrow_, ncol_, brow_, bcol_, expected_density, this->nnzb,
                this->indptr, this->indices, seed_);
    this->density = (float)this->nnzb / (nrow_/brow_) / (ncol_/bcol_);

    if (row_permute_) {
        std::vector<int> permutation(nrow_);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(),
                    std::default_random_engine(seed_));
        this->row_permute_ids = permutation;
    }
    else this->row_permute_ids.resize(0);

    // generate data
    this->values.resize(this->nnzb * brow_ * bcol_);
    for (auto it = this->values.begin(); it != this->values.end(); it++) {
        //*it = static_cast<T>((float)(std::rand() % 9 - 4));
        *it = static_cast<T>((float)(1.0f));
    }

    // generate a config string for logging
    std::stringstream s;
    s << nrow_ << "," << ncol_ << "," << brow_ << "," << bcol_ << ","
        << this->density << "," << seed_ << ",";
    if (this->row_permute) {
        s << "row-permute";
    }
    else {
        s << "non-row-permute";
    }
    config_str = s.str();

    // set flag
    initialized = true;
    if (device_synced) {
        // clear any old version
        this->__clear_device_ref();
        device_synced = false;
    }
}

template<typename T>
void BlockwiseSpMatrix<T>::init_sparse(int nrow_, int ncol_, int brow_, int bcol_, int nnz_, bool row_permute_, const std::vector<int>& hA_cvsOffsets, const std::vector<int>& hA_columns, std::vector<T>& hA_values, unsigned seed_){

    assert(nrow_ % brow_ == 0);
    assert(ncol_ % bcol_ == 0);

    this->nrow = nrow_;
    this->ncol = ncol_;
    this->brow = brow_;
    this->bcol = bcol_;
    this->row_permute = row_permute_;
    this->nnzb = nnz_;
    this->seed = seed_;

    this->indptr = hA_cvsOffsets;
    this->indices = hA_columns;
    this->values = hA_values;

    this->density = (float)this->nnzb / (nrow_/brow_) / (ncol_/bcol_);

    if (row_permute_) {
        std::vector<int> permutation(nrow_);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(),
                    std::default_random_engine(seed_));
        this->row_permute_ids = permutation;
    }
    else this->row_permute_ids.resize(0);

    // generate data
    /* this->values.resize(this->nnzb * brow_ * bcol_);
    for (auto it = this->values.begin(); it != this->values.end(); it++) {
        //*it = static_cast<T>((float)(std::rand() % 9 - 4));
        *it = static_cast<T>((float)(1.0f));
    } */

    // generate a config string for logging
    std::stringstream s;
    s << nrow_ << " " << ncol_ << " " << brow_ << " " << bcol_ << " "
        << this->density << " " << seed_ << " ";
    if (this->row_permute) {
        s << "row-permute ";
    }
    else {
        s << "non-row-permute ";
    }
    config_str = s.str();

    // set flag
    initialized = true;
    if (device_synced) {
        // clear any old version
        this->__clear_device_ref();
        device_synced = false;
    }
}

template<typename T>
void BlockwiseSpMatrix<T>::transform_and_sync_device() {

    assert(initialized && "must call initialize() before sync to device.\n");

    // create host format
    int size_of_meta = 4*(
        (ROUND_UP(indptr.size(), alignment))
        + (ROUND_UP(indices.size(), alignment))
        + (ROUND_UP(row_permute_ids.size(), alignment)));
    meta_h = std::vector<char>(size_of_meta);
    memset(meta_h.data(), 0x0, size_of_meta*sizeof(char));

    int offset = 0;
    memcpy(meta_h.data(), indptr.data(), sizeof(int)*indptr.size());
    offset += indptr.size();
    while (offset % alignment != 0) offset++;

    std::vector<int> ind_tmp = indices;
    for (auto it = ind_tmp.begin(); it != ind_tmp.end(); it++)
    {    *it *= bcol; }

    memcpy(meta_h.data() + offset * (sizeof(int)/sizeof(char)),
            ind_tmp.data(), sizeof(int)*ind_tmp.size());
    offset += ind_tmp.size();
    while (offset % alignment != 0) offset++;

    memcpy(meta_h.data() + offset * (sizeof(int)/sizeof(char)),
            row_permute_ids.data(), sizeof(int)*row_permute_ids.size());

    data_h = values;

    // allocate device memory
    void *meta_d; T* data_d;
    size_t meta_size = meta_h.size() *sizeof(char);
    size_t data_size = data_h.size() *sizeof(T);

    CUDA_CHECK( cudaMalloc(&meta_d, meta_size));
    CUDA_CHECK( cudaMalloc(&data_d, data_size));

    CUDA_CHECK( cudaMemcpy(meta_d, meta_h.data(),
                    meta_size, cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(data_d, data_h.data(),
                    data_size, cudaMemcpyHostToDevice));

    device_ref._meta = meta_d;
    device_ref._data = data_d;

    offset = 0;
    device_ref.csb_indptr = (int*)meta_d;

    offset += indptr.size();
    while (offset % alignment != 0) offset++;
    device_ref.csb_indices = (int*)meta_d + offset;

    if (row_permute_ids.size() > 0) {
        offset += indices.size();
        while (offset % alignment != 0) offset++;
        device_ref.row_permute_ids = (int*)meta_d + offset;
    }
    else device_ref.row_permute_ids= nullptr;

    device_ref.csb_values = data_d;

    // set flag
    device_synced = true;
}

template<typename T>
void get_host_reference(BlockwiseSpMatrix<T> &spmat, const int N,
const std::vector<T> &B, float alpha, const std::vector<T> &C, float beta,
std::vector<float> &D_ref) {
    int brow = spmat.brow;
    int bcol = spmat.bcol;
    int M    = spmat.nrow;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N+j;
            D_ref[idx] = 0.f;
        }
    }

    for (int i = 0; i < M/brow; i++) {
        int begin = spmat.indptr[i];
        int end   = spmat.indptr[i+1];
        auto itA = spmat.values.begin() + begin * brow * bcol;
        for (int p = begin; p < end; p++, itA += brow*bcol) {
            int j = spmat.indices[p] * bcol;
            auto itB = B.begin() + j * N;
            for (int kk = 0; kk < bcol; kk++) {
                for (int mm = 0; mm < brow; mm++) {
                    for (int nn = 0; nn < N; nn++) {
                        int idx_D = (i * brow + mm) *N + nn;
                        if (spmat.row_permute) {
                            idx_D = spmat.row_permute_ids[i*brow + mm]*N + nn;
                        }
                        float a = static_cast<float>(*(itA + kk*brow + mm));
                        float b = static_cast<float>(*(itB + kk*N  + nn));
                        D_ref[idx_D] += (a*b);
    }}}}}

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N+j;
            float d = D_ref[idx];
            float c = static_cast<float>(C[idx]);
            d = d * alpha + c * beta;
            D_ref[idx] = d;
        }
    }
}
