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

#include "format.hpp"

template<class T> Format<T>::Format(){
    this->A_num_rows=0;
    this->A_num_cols=0;
    this->A_size=0;
    this->A_nnz=0;
    this->seed=2022;
    this->density=0.0f;
}

template<class T> Format<T>::Format(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_){
    this->A_num_rows=nrow_;
    this->A_num_cols=ncol_;
    this->A_size=nrow_*ncol_;
    this->A_nnz = nnz_;
    this->density=density_;
    this->seed=seed_;
}

template<class T> Format<T>::~Format() = default;

template<class T> int Format<T>::get_A_num_cols(){
    return A_num_cols;
}

template<class T> int Format<T>::get_A_num_rows(){
    return A_num_rows;
}

template<class T>
int Format<T>::get_A_nnz(){
    return A_nnz;
}

template class Format<float>;
template class Format<__half>;
template class Format<int8_t>;