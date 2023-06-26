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

namespace spatha {

template<typename T>
struct BlockwiseSpMatrix {
    // shape and pattern configs
    int nrow;
    int ncol;
    int nnzb;
    int brow;
    int mbrow;
    int n;
    int m;
    int bm;
    float density;
    unsigned seed;

    int A_num_cols_sp;
    int A_num_cols_sp_pad_nm;
    int A_num_cols_sp_pad;

    // !IMPORTANT! value fixed because of NVIDIA architecture (2:4)
    int m_fixed, mrow_m, nelems, nelems_col;
    int bits_elem_meta, bits_elem_cols;
    int brow_fixed;

    // original format on host
    const int alignment = 8;
    std::vector<uint> metadata;
    std::vector<uint> indices;
    std::vector<T> values;

    // transformed format on host
    std::vector<char> meta_h;
    std::vector<T>    data_h;

    // pointers to transformed format on device
    struct DeviceRef {
        void *_meta = nullptr;
        T    *_data = nullptr;
        T    *values;
        uint *metadata;
        uint *csb_indices;
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

    //void random_mask(int nrow, int ncol, int n, int m, int mbrow, float density, std::vector<unsigned int> &metadata, unsigned seed);
    void random_mask(int nrow, int ncol, int ncol_pad, int n, int m, int brow, float density, std::vector<unsigned int> &metadata, std::vector<unsigned int> &indices, unsigned seed, int bm_);

    //void init_random(int nrow_, int ncol_, int brow_, int n_, int m_, float density_, unsigned seed_, int mbrow_=1);
    void init_random(int nrow_, int ncol_, int brow_, int n_, int m_,
    int density_, unsigned seed_, int mbrow_=1, int bm_=1);

    void init_sparse(int nrow_, int ncol_, int brow_, int n_, int m_,
    int nnz_, const std::vector<uint>& hA_metadata, std::vector<T>& hA_values, const std::vector<uint>& hA_indices, unsigned seed_, int mbrow_=1, int bm_=1);

    void init_sparse_device(int nrow_, int ncol_, int brow_, int n_, int m_,
    int nnz_, int* hA_metadata, T* hA_values, int* hA_indices, unsigned seed_, int mbrow_=1, int bm_=1);

    void transform_and_sync_device();
};


template<typename T>
void BlockwiseSpMatrix<T>::init_random(int nrow_, int ncol_, int brow_, int n_, int m_, int density_, unsigned seed_, int mbrow_, int bm_){

    assert(nrow_ % brow_ == 0);
    //assert(ncol_ % bcol_ == 0);

    this->nrow = nrow_;
    this->ncol = ncol_;
    this->brow = brow_;
    this->mbrow = mbrow_;
    this->n = n_;
    this->m = m_;
    this->seed = seed_;
    this->bm = bm_;
    density = (float)n_/(float)m_;
    // !IMPORTANT! constants because of architecture constraints
    m_fixed = 4;
    bits_elem_meta=2;
    mrow_m = 2;
    bits_elem_cols=8;
    brow_fixed = 16;
    nelems=(sizeof(uint)*8)/bits_elem_meta;
    nelems_col = nelems/mrow_m;

    A_num_cols_sp = (ncol_/m)*n;
    A_num_cols_sp_pad_nm = (ROUND_UP(this->ncol, m)/m)*n;
    A_num_cols_sp_pad = ROUND_UP((ROUND_UP(ncol_, m)/m)*n, 16); //16=mma_K/2
    this->nnzb = this->nrow*A_num_cols_sp_pad;

    this->values.resize(this->nnzb, 0);
    this->indices.resize(this->nrow/bm_ * A_num_cols_sp_pad/n*m_fixed, 0);//4=m_fixed
    this->metadata.resize(this->nrow/mrow_m * A_num_cols_sp_pad/nelems_col, 0);

    BlockwiseSpMatrix<T>::random_mask(nrow_, A_num_cols_sp, A_num_cols_sp_pad_nm, n_, m_, mbrow_, density, this->metadata, this->indices, seed_, bm_);

    for (auto it = this->values.begin(); it != this->values.end(); it++) {
        *it = static_cast<T>((float)(std::rand() % 9 - 4));
        //*it = static_cast<T>((float)(std::rand() % 8 + 1));
    }

    std::stringstream s;
    s << nrow_ << "," << ncol_ << "," << brow_ << "," << n_ << "," << m_ << ","
        << density << "," << seed_ << ",";
    config_str = s.str();

    this->initialized = true;
    if (this->device_synced) {
        this->__clear_device_ref();
        this->device_synced = false;
    }
}

template<typename T>
void BlockwiseSpMatrix<T>::random_mask(int nrow_, int ncol_, int ncol_pad, int n, int m, int mbrow, float density, std::vector<uint> &metadata, std::vector<uint> &indices, unsigned seed, int bm_)
{
    std::vector<unsigned int> arr;
    for(int i = 0; i < m_fixed; i++){ arr.push_back(i); }

    // metadata
    for(int i=0; i<nrow_/mbrow; i++){
        int j=0;
        for(; j<ncol_/nelems_col; j++){
            for(int k=0; k<mbrow/mrow_m; k++){
                unsigned int meta=0;
                for(int g=0; g<nelems/n; g++){
                    random_shuffle(arr.begin(), arr.end());
                    std::sort(arr.begin(), arr.begin() + n);

                    for(int w=0; w<n; w++){
                        unsigned int tmp = arr[w];

                        meta |= (tmp << (g*n*bits_elem_meta+w*bits_elem_meta));
                    }
                }
                metadata[
                        i*mbrow/mrow_m*A_num_cols_sp_pad/nelems_col+
                        j*mbrow/mrow_m+
                        k] = meta;
            }
        }

        if(ncol_pad>ncol_){
            for(int k=0; k<mbrow/mrow_m; k++){
                unsigned int meta=0;
                int resto = (this->ncol)-(4*m*(ncol_/nelems_col));
                for(int t=0; t<mrow_m; t++){
                    int g=0;

                    for(; g<resto/m; g++){
                        random_shuffle(arr.begin(), arr.end());
                        std::sort(arr.begin(), arr.begin() + n);

                        for(int w=0; w<n; w++){
                            unsigned int tmp = arr[w];

                            meta |= (tmp << ((g+t*4)*n*bits_elem_meta+w*bits_elem_meta));
                        }
                    }

                    if(resto%m>0){
                        for(int w=0; w<n; w++){
                            unsigned int tmp = w;

                            meta |= (tmp << ((g+t*4)*n*bits_elem_meta + w*bits_elem_meta));
                        }
                    }
                }

                metadata[i*mbrow/2*A_num_cols_sp_pad/8+
                            j*mbrow/2+
                            k] = meta;
            }
        }
    }

    std::vector<unsigned int> arr2;
    for(int i = 0; i < m; i++){ arr2.push_back(i); }

    int resto = this->ncol-(ncol_/n)*m;
    for(int i=0; i<(nrow_/bm_); i++){
        int j=0;
        for(; j<ncol_/n; j++){
            random_shuffle(arr2.begin(), arr2.end());
            std::sort(arr2.begin(), arr2.begin() + m_fixed);

            for(int w=0; w<m_fixed; w++){
                uint tmp = arr2[w];

                this->indices[i*(A_num_cols_sp_pad/n)*m_fixed +
                                 j*m_fixed + w] = tmp;
            }
        }
        if(resto>0){
            for(int w=0; w<m_fixed; w++){
                this->indices[i*(A_num_cols_sp_pad/n)*m_fixed +
                                    j*m_fixed + w] = w;
            }
        }
    }
}

template<typename T>
void BlockwiseSpMatrix<T>::init_sparse(int nrow_, int ncol_, int brow_, int n_, int m_, int nnz_, const std::vector<unsigned int>& hA_metadata, std::vector<T>& hA_values, const std::vector<unsigned int>& hA_indices, unsigned seed_, int mbrow_, int bm_){

    assert(nrow_ % brow_ == 0);
    //assert(ncol_ % bcol_ == 0);

    this->nrow = nrow_;
    this->ncol = ncol_;
    this->brow = brow_;
    this->mbrow = mbrow_;
    this->n = n_;
    this->m = m_;
    this->nnzb = nnz_;
    this->seed = seed_;
    this->bm = bm_;

    this->metadata = hA_metadata;
    this->indices  = hA_indices;
    this->values   = hA_values;

    this->density = (float)n_ / (float)m_;

    // generate a config string for logging
    std::stringstream s;
    s << nrow_ << "," << ncol_ << "," << brow_ << "," << n_ << "," << m_ << ","
        << this->density << "," << seed_ << ",";
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
void BlockwiseSpMatrix<T>::init_sparse_device(int nrow_, int ncol_, int brow_, int n_, int m_, int nnz_, int* hA_metadata, T* hA_values, int* hA_indices, unsigned seed_, int mbrow_, int bm_){

    assert(nrow_ % brow_ == 0);
    //assert(ncol_ % bcol_ == 0);

    this->nrow = nrow_;
    this->ncol = ncol_;
    this->brow = brow_;
    this->mbrow = mbrow_;
    this->n = n_;
    this->m = m_;
    this->nnzb = nnz_;
    this->seed = seed_;
    this->bm = bm_;

    //printf("init sparse A[0]:%.2f \n", __half2float(hA_values[0]));

    this->device_ref.metadata    = reinterpret_cast<uint*>(hA_metadata);
    this->device_ref.csb_indices = reinterpret_cast<uint*>(hA_indices);
    this->device_ref.values      = hA_values;

    device_synced = true;

    this->density = (float)n_ / (float)m_;

    // generate a config string for logging
    std::stringstream s;
    s << nrow_ << "," << ncol_ << "," << brow_ << "," << n_ << "," << m_ << ","
        << this->density << "," << seed_ << ",";
    config_str = s.str();

    // set flag
    initialized = true;
    device_synced = true;
    /* if (device_synced) {
        // clear any old version
        this->__clear_device_ref();
        device_synced = false;
    } */
}

template<typename T>
void BlockwiseSpMatrix<T>::transform_and_sync_device() {

    assert(this->initialized && "must call initialize() before sync to device.\n");

    // create host format
    int size_of_meta = sizeof(uint)*(ROUND_UP(this->metadata.size(), this->alignment)+ ROUND_UP(this->indices.size(), this->alignment));
    this->meta_h = std::vector<char>(size_of_meta);
    memset(this->meta_h.data(), 0x0, size_of_meta*sizeof(char));

    int offset = 0;
    memcpy(this->meta_h.data(), this->metadata.data(), sizeof(uint)*this->metadata.size());
    offset+=this->metadata.size();
    while(offset&this->alignment != 0) offset++;

    memcpy(this->meta_h.data() + offset * (sizeof(uint)/sizeof(char)), this->indices.data(), sizeof(uint)*this->indices.size());

    this->data_h = this->values;

    // allocate device memory
    void *meta_d; T* data_d;
    size_t meta_size = this->meta_h.size() *sizeof(char);
    size_t data_size = this->data_h.size() *sizeof(T);

    CUDA_CHECK( cudaMalloc(&meta_d, meta_size));
    CUDA_CHECK( cudaMalloc(&data_d, data_size));

    CUDA_CHECK( cudaMemcpy(meta_d, this->meta_h.data(),
                    meta_size, cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(data_d, this->data_h.data(),
                    data_size, cudaMemcpyHostToDevice));

    this->device_ref._meta = meta_d;
    this->device_ref._data = data_d;

    offset=0;
    this->device_ref.metadata = (uint*) meta_d;

    offset += this->metadata.size();
    while(offset % this->alignment != 0) offset++;
    this->device_ref.csb_indices = (uint*) meta_d + offset;

    this->device_ref.values = data_d;

    // set flag
    device_synced = true;
}

}