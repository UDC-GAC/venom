
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

#include <iostream>
#include "dataset.hpp"

template <class T, class T2>
Dataset<T,T2>::Dataset(Format<T> &format_): format(format_){
};

template <class T, class T2>
Dataset<T,T2>::Dataset(int m, int k, int n, float density, Format<T>& format, int seed, int block_sz)
:format(format){
    this->A_num_rows = m;
    this->A_num_cols = k;
    this->B_num_rows = k;
    this->B_num_cols = n;
    this->A_size = m*k;
    this->B_size = k*n;
    this->C_size = m*n;

    B.init_matrix(B_size);
    C.init_matrix(C_size);

    //std::cout << m << " " << k << " " << n << std::endl;

    hB = B.host_ptr;
    hC = C.host_ptr;

    this->format.init(m, k, 0, density, seed, false, block_sz, 1);
};

template <class T, class T2>
Dataset<T,T2>::Dataset(int m, int k, int n, float density, Format<T>& format, int seed, int brow_, int bcol_)
:format(format){
    this->A_num_rows = m;
    this->A_num_cols = k;
    this->B_num_rows = k;
    this->B_num_cols = n;
    this->A_size = m*k;
    this->B_size = k*n;
    this->C_size = m*n;

    B.init_matrix(B_size);
    C.init_matrix(C_size);

    //std::cout << m << " " << k << " " << n << std::endl;

    hB = B.host_ptr;
    hC = C.host_ptr;

    this->format.init(m, k, 0, density, seed, false, brow_, bcol_);
};

template <class T, class T2>
Dataset<T,T2>::Dataset(int a_m, int a_k, int b_n, float density, Format<T> &format, int seed, int n_, int m_, int brow_, int mbrow_, int bm_)
:format(format){
    this->A_num_rows = a_m;
    this->A_num_cols = a_k;
    this->B_num_rows = a_k;
    this->B_num_cols = b_n;
    this->A_size = a_m*a_k;
    this->B_size = a_k*b_n;
    this->C_size = a_m*b_n;

    //B.init_matrix(B_size);
    B.init_matrix(B_size, ROUND_UP(a_k, 2)*b_n);
    C.init_matrix(C_size);

    //std::cout << a_m << " " << a_k << " " << b_n << std::endl;

    hB = B.host_ptr;
    hC = C.host_ptr;

    this->format.init(a_m, a_k, n_, density, seed, false, m_, brow_, mbrow_, bm_);
}

template <class T, class T2>
Dataset<T,T2>::~Dataset<T,T2>() = default;

template <typename T, typename T2> void Dataset<T,T2>::set_B(T value){
    for (size_t i = 0; i < B_size; i++)
    {
        hB.push_back(value);
    }
}

template <typename T, typename T2> void Dataset<T,T2>::set_C(T2 value){
    for (size_t i = 0; i < C_size; i++)
    {
        hC.push_back(value);
    }
}

template <class T, class T2>
Format<T>* Dataset<T,T2>::get_format(){
    return &format;
}

template <typename T, typename T2>
vector<T>& Dataset<T,T2>::get_B(){
    return hB;
}

template <typename T, typename T2>
vector<T2>& Dataset<T,T2>::get_C(){
    return hC;
}

template <typename T, typename T2>
Matrix<T>& Dataset<T,T2>::get_Bmat(){
    return B;
}

template <typename T, typename T2>
Matrix<T2>& Dataset<T,T2>::get_Cmat(){
    return C;
}


template <typename T, typename T2>
int Dataset<T,T2>::get_A_size(){
    return A_size;
}

template <typename T, typename T2>
int Dataset<T,T2>::get_B_size(){
    return B_size;
}

template <typename T, typename T2>
int Dataset<T,T2>::get_C_size(){
    return C_size;
}


template <typename T, typename T2>
int Dataset<T,T2>::get_B_num_rows(){
    return A_num_cols;
}

template <typename T, typename T2>
int Dataset<T,T2>::get_B_num_cols(){
    return B_num_cols;
}

template <typename T, typename T2>
int Dataset<T,T2>::get_A_num_cols(){
    return A_num_cols;
}

template <typename T, typename T2>
int Dataset<T,T2>::get_A_num_rows(){
    return A_num_rows;
}

template class Dataset<int8_t, int8_t>;
template class Dataset<float, float>;
template class Dataset<__half, __half>;