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

#include "format_cvs.hpp"

template<class T>
Format_cvs<T>::~Format_cvs() = default;

template<class T>
void Format_cvs<T>::init(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_, bool row_permute_, int brow_, int bcol_, int mbrow, int bm){

    Format_cxx<T>::init(nrow_, ncol_, nnz_, density_, seed_, row_permute_, brow_, bcol_);

    // generate a config string for logging
    std::stringstream s;
    s << nrow_ << " " << ncol_ << " " << brow_ << " " << bcol_ << " "
        << density_ << " " << seed_ << " ";
    if (row_permute_) {
        s << "row-permute ";
    }
    else {
        s << "non-row-permute ";
    }

    //this->config_str = s.str();
}

template<class T>
int Format_cvs<T>::get_vec_length(){
    return this->brow;
}

template<class T>
void Format_cvs<T>::to_String(){
    std::cout << "v_length: " << this->brow << std::endl;
    for (auto i = this->hA_Offsets.begin(); i != this->hA_Offsets.end(); ++i)
        std::cout << *i << " ";
    std::cout << std::endl;
    for (auto i = this->hA_columns.begin(); i != this->hA_columns.end(); ++i)
        std::cout << *i << " ";
    std::cout << std::endl << std::endl;
    for (auto i = this->hA_values.begin(); i != this->hA_values.end(); ++i)
        std::cout << static_cast<float>(*i) << " ";
    std::cout << std::endl << std::endl;
}

template<class T>
std::vector<T>& Format_cvs<T>::to_dense(){
    int col;
    T val;
    int r_prev = 0;
    int r_num = 0;
    int r;

    for (size_t i = 0; i < this->A_size; i++)
    {
        this->hA_dense.push_back(0.0f);
    }

    for (size_t i = 1; i < this->A_num_rows/this->brow+1; i++)
    {
        r = this->hA_Offsets[i];
        if(r!=r_prev && r!=0){
            //std::cout << r_prev << " " << r << std::endl;
            for (size_t c = r_prev; c < r; c++)
            {
                col = this->hA_columns[c];
                val = this->hA_values[c];
                for(int v=0; v<this->brow; v++){
                    this->hA_dense[(r_num+v)*this->A_num_cols + col] = val;
                }
            }

        }
        r_prev = r;
        r_num+=this->brow;
    }

    return this->hA_dense;
}

template<class T>
void Format_cvs<T>::change_v_length(int wm){
    //assert(wm > v_length || v_length%wm);

    int offset = this->brow/wm;

    std::vector<int> indptr_tmp((this->hA_Offsets.size()-1)*offset + 1);
    std::vector<int> indices_tmp(this->hA_columns.size()*offset);
    std::vector<T>   values_tmp(this->hA_values.size());

    indptr_tmp[0] = 0;
    int acum=0;
    int idx_prev = 0;
    for(auto i=1; i<this->hA_Offsets.size(); i++){
        int idx = this->hA_Offsets[i];
        for(auto j=0; j<offset; j++){
            acum+=(idx-idx_prev);
            indptr_tmp[(i-1)*offset+j+1] = acum;
        }
        idx_prev = idx;
    }

    acum=0;
    int tmp=0;
    for(auto k=1; k<this->hA_Offsets.size(); k++){
        int cols = this->hA_Offsets[k];
        for(auto j=0; j<offset; j++){
            tmp=0;
            for(auto i=acum; i<cols; i++){
                indices_tmp[acum*offset+j*(cols-acum)+tmp] = this->hA_columns[i];
                tmp++;
            }
        }
        acum=cols;
    }


    acum=0;
    for(auto i=0; i<this->hA_Offsets.size(); i++){
        int cols = this->hA_Offsets[i];
        for(auto j=0; j<offset; j++){
            for(auto k=0; k<cols-acum; k++){
                for(auto m=0; m<wm; m++){
                    values_tmp[acum*this->brow+j*wm*(cols-acum)+k*wm+m] = this->hA_values[acum*this->brow+j*wm+k*this->brow+m];
                }
            }
        }
        acum=cols;
    }

    // replace values
    this->brow = wm;

    this->hA_Offsets.resize(indptr_tmp.size(), 0);
    for(size_t i=0; i<indptr_tmp.size(); i++){
        this->hA_Offsets[i] = indptr_tmp[i];
    }

    this->hA_columns.resize(indices_tmp.size(), 0);
    for(size_t i=0; i<indices_tmp.size(); i++){
        this->hA_columns[i] = indices_tmp[i];
    }

    this->hA_values.resize(values_tmp.size(), 0);
    for(size_t i=0; i<values_tmp.size(); i++){
        this->hA_values[i] = values_tmp[i];
    }
}


template class Format_cvs<float>;
template class Format_cvs<__half>;
template class Format_cvs<int8_t>;