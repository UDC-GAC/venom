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
#include <vector>
#include "cuda_error.h"      // CUDA_CHECK

namespace spatha {

template<typename T>
struct CudaArray {
    int             size = 0;
    std::vector<T>  host_array;
    T*              device_ptr = nullptr;
    bool           initialized = false;
    bool         device_synced = false;
    virtual void __fill(){}
    void initialize(int size_) {
        this->size=size_;
        this->host_array.clear();
        this->host_array.resize(size_);
        if (size_ > 0) {
            this->__fill();
            CUDA_CHECK( cudaMalloc(&this->device_ptr, size_*sizeof(T)));
            this->sync_device();
        }
        initialized = true;
    }

    void initialize(const std::vector<T> &that) {
        host_array = that;
        size       = that.size();
        if (size > 0) {
            this->__fill();
            CUDA_CHECK( cudaMalloc(&this->device_ptr, size*sizeof(T)));
            this->sync_device();
        }
        initialized = true;
    }

    virtual ~CudaArray() {
        if (device_ptr) {
            CUDA_CHECK( cudaFree(device_ptr));
            device_ptr = nullptr;
        }
    }

    void sync_device() {
        if (size == 0) return;
        CUDA_CHECK( cudaMemcpy(this->device_ptr, this->host_array.data(),
            this->size*sizeof(T), cudaMemcpyHostToDevice));
        device_synced = true;
    }

    void sync_host() {
        if (size == 0) return;
        CUDA_CHECK( cudaMemcpy(this->host_array.data(), this->device_ptr,
            this->size*sizeof(T), cudaMemcpyDeviceToHost));
    }
    std::vector<T>& host_ref() {
        return this->host_array;
    }
    T* device_data() {
        return this->device_ptr;
    }
};


template<typename T>
struct CudaRandomArray: public CudaArray<T> {
public:
    void __fill() {
        for (int i = 0; i < this->size; i++) {
            //this->host_array[i] = static_cast<T>((float)(std::rand() % 9 - 4));
            this->host_array[i] = static_cast<T>((float)(std::rand() % 8 + 1));
        }
    }
};

template<typename T>
struct CudaZerosArray: public CudaArray<T> {
public:
    void __fill() {
        for (int i = 0; i < this->size; i++) {
            this->host_array[i] = static_cast<T>(0.0f);
        }
    }
};

template<typename T>
struct CudaOnesArray: public CudaArray<T> {
public:
    void __fill() {
        for (int i = 0; i < this->size; i++) {
            this->host_array[i] = static_cast<T>(1.0f);
        }
    }
};

// struct CudaBalancedMetaArrayB16 : public CudaArray<uint32_t> {
//     int nrow;
//     int ncol;
// public:
//     std::vector<int> unpacked_mask;
//     void __fill() {
//         generateBalancedMaskB16(nrow, ncol, unpacked_mask);
//         convertMaskToMetaB16(nrow, ncol, unpacked_mask, this->host_array);
//     }
//     void initialize(int nrow_, int ncol_) {
//         this->nrow = nrow_;
//         this->ncol = ncol_;
//         this->unpacked_mask = std::vector<int>(nrow*ncol, 0);
//         int size = (CEIL(nrow_, MetaPackX))*(CEIL(ncol_, MetaPackY_B16))
//                      * MetaPackSz;
//         CudaArray<uint32_t>::initialize(size); // internally call 'fill()'
//     }
// };

// struct CudaBalancedMetaArrayB32 : public CudaArray<uint32_t> {
//     int nrow;
//     int ncol;
// public:
//     std::vector<int> unpacked_mask;
//     void __fill() {
//         generateBalancedMaskB32(nrow, ncol, unpacked_mask);
//         convertMaskToMetaB32(nrow, ncol, unpacked_mask, this->host_array);
//     }
//     void initialize(int nrow_, int ncol_) {
//         this->nrow = nrow_;
//         this->ncol = ncol_;
//         this->unpacked_mask = std::vector<int>(nrow*ncol, 0);
//         int size = (CEIL(nrow_, MetaPackX))*(CEIL(ncol_, MetaPackY_B16))
//                      * MetaPackSz;
//         CudaArray<uint32_t>::initialize(size); // internally call 'fill()'
//     }
// };

}