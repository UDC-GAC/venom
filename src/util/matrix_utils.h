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

#include <vector>
#include <iostream>
#include <stdio.h>
#include "utils.h"

template<typename T>
struct Matrix {
    bool init = false;
    int size=0;
    int size_np=0;
    std::vector<T> host_ptr;
    T* device_ptr = nullptr;
    bool device_synced = false;
    virtual void __init_matrix(){}

    void init_matrix(const std::vector<T> &matrix_) {
        host_ptr = matrix_;
        size = matrix_.size();
        if (size > 0) {
            this->__init_matrix();
            CUDA_CHECK( cudaMalloc(&this->device_ptr, size*sizeof(T)));
            this->sync_device();
        }
        init = true;
    }

    void init_matrix(int size_, int pad=0) {
        this->size_np=size_;
        this->size=size_+pad;
        this->host_ptr.clear();
        this->host_ptr.resize(this->size, 0);
        if (this->size > 0) {
            this->__init_matrix();
            CUDA_CHECK( cudaMalloc(&this->device_ptr, this->size*sizeof(T)));
            this->sync_device();
        }
        init = true;
    }

    void sync_host() {
        if (!size) return;
        CUDA_CHECK( cudaMemcpy(this->host_ptr.data(), this->device_ptr,
            this->size*sizeof(T), cudaMemcpyDeviceToHost));
    }

    void sync_device() {
        if (!size) return;
        CUDA_CHECK( cudaMemcpy(this->device_ptr, this->host_ptr.data(),
            this->size*sizeof(T), cudaMemcpyHostToDevice));
        device_synced = true;
    }

    virtual ~Matrix() {
        if (device_ptr) {
            CUDA_CHECK( cudaFree(device_ptr));
            device_ptr = nullptr;
        }
    }
};

template<typename T>
struct ZerosMatrix: public Matrix<T> {
public:
    void __init_matrix() {
        for (int i = 0; i < this->size_np; i++) {
            this->host_ptr[i] = static_cast<T>(0.0f);
        }
    }
};

template<typename T>
struct RandomMatrix: public Matrix<T> {
public:
    void __init_matrix() {
        for (int i = 0; i < this->size_np; i++) {
            //this->host_ptr[i] =  static_cast <T> (rand()/RAND_MAX);
            this->host_ptr[i] = static_cast<T>((float)(std::rand() % 9 - 4));
            //this->host_ptr[i] = static_cast<T>((float)(std::rand() % 8 + 1));
        }
    }
};

template<typename T>
struct OnesMatrix: public Matrix<T> {
public:
    void __init_matrix() {
        for (int i = 0; i < this->size_np; i++) {
            this->host_ptr[i] = static_cast<T>(1.0f);
        }
    }
};