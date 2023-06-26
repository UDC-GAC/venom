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

#include "format_csr.hpp"

template<class T>
Format_csr<T>::~Format_csr() = default;

template<class T>
void Format_csr<T>::init(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_, bool row_permute_, int brow_, int bcol_, int mbrow, int bm_){
    Format_cxx<T>::init(nrow_, ncol_, nnz_, density_, seed_, row_permute_, 1, 1);

    // generate a config string for logging
    std::stringstream s;
    s << nrow_ << " " << ncol_ <<  " " << density_ << " " << seed_ << " ";
    if (row_permute_) {
        s << "row-permute ";
    }
    else {
        s << "non-row-permute ";
    }
    //this->config_str = s.str(); //FIXME:
}

template<class T>
void Format_csr<T>::to_String(){
    for (auto i = this->hA_Offsets.begin(); i != this->hA_Offsets.end(); ++i)
        std::cout << *i << " ";
    std::cout << std::endl;
    for (auto i = this->hA_columns.begin(); i != this->hA_columns.end(); ++i)
        std::cout << *i << " ";
    std::cout << std::endl << std::endl;
}

template<class T>
std::vector<T>& Format_csr<T>::to_dense(){
    int col;
    T val;
    int r_prev = 0;
    int r_num = 0;
    int r;

    for (size_t i = 0; i < this->A_size; i++)
    {
        this->hA_dense.push_back(0.0f);
    }

    for (size_t i = 1; i < this->A_num_rows+1; i++)
    {
        r = this->hA_Offsets[i];
        if(r!=r_prev && r!=0){
            for (size_t c = r_prev; c < r; c++)
            {
                col = this->hA_columns[c];
                val = this->hA_values[c];
                this->hA_dense[r_num*this->A_num_cols + col] = val;
            }

        }
        r_prev = r;
        r_num++;
    }

    return this->hA_dense;
}


template class Format_csr<float>;
template class Format_csr<__half>;
template class Format_csr<int8_t>;