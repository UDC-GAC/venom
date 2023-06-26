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

#include "spmm.hpp"

template<class T, class T2>
Spmm_CNN<T,T2>::Spmm_CNN(Dataset<T,T2> &d)
    :dset(d)
{
    cudaMalloc((void**) &dB, d.get_B_size() * sizeof(T));
    cudaMalloc((void**) &dC, d.get_C_size() * sizeof(T2));
}

template<class T, class T2>
Spmm_CNN<T,T2>::~Spmm_CNN(){
    cudaFree(dB);
    cudaFree(dC);
}

template<class T, class T2>
T* Spmm_CNN<T,T2>::get_dB(){
    return dB;
}

template<class T, class T2>
T2* Spmm_CNN<T,T2>::get_dC(){
    return dC;
}

template<class T, class T2>
Dataset<T,T2>& Spmm_CNN<T,T2>::get_dataset(){
    return dset;
}

template class Spmm_CNN<float, float>;
template class Spmm_CNN<__half, __half>;
template class Spmm_CNN<int8_t, int8_t>;