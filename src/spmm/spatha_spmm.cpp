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

#include "./spatha_spmm.hpp"

template<class T, class T2>
Spatha<T,T2>::Spatha(Format_sr_nm<T>& f, Dataset<T,T2> &d, cudaDataType_t S)
:Spmm_CNN<T,T2>(d){}

template<class T, class T2>
Spatha<T,T2>::Spatha(Dataset<T,T2> &d, cudaDataType_t S, cusparseComputeType C_spmm)
:Spmm_CNN<T,T2>(d){
    Format_sr_nm<T> *f = dynamic_cast<Format_sr_nm<T>*>(d.get_format());

    int A_num_rows = f->get_A_num_rows();
    int A_num_cols = f->get_A_num_cols();

    this->spmat.init_sparse(A_num_rows, A_num_cols, f->get_vec_length(), f->get_n(), f->get_m(), f->get_A_nnz(), f->get_hA_metadata(), f->get_hA_val(), f->get_hA_indices(), 2022, f->get_meta_vec_length(), f->get_bm());

    d.get_Bmat().sync_device();
    d.get_Cmat().sync_device();
    this->spmat.transform_and_sync_device();

    this->gflop_count = (float)A_num_rows / 1e9 * (d.get_B_num_cols())*A_num_cols*2;
}

template<class T, class T2>
Spatha<T,T2>::~Spatha() = default;

template<class T, class T2>
std::vector<T2>& Spatha<T,T2>::get_result(){
    return result;
}

template<class T, class T2>
std::vector<T>& Spatha<T,T2>::get_sparse(){
    std::cerr << "Operation not supported yet \n";
    exit(EXIT_FAILURE);
}

template<class T, class T2>
float Spatha<T,T2>::spmm(int times, int bm_, int bn_, int bk_, int wm_, int wn_, int wk_, int mm_, int mn_, int mk_, int nstage_){
    Dataset<T,T2>& d = this->get_dataset();
    cudaStream_t stream = NULL;
    float time=0.0f;

    int m=d.get_A_num_rows(), n=d.get_B_num_cols(), k=d.get_A_num_cols();

    spatha::SpmmNMInitFn_t init_func = init_fn[std::make_tuple(bm_, bn_, bk_, wm_, wn_, wk_, mm_, mn_, mk_, nstage_)];
    spatha::SpmmNMExecFn_t exec_func = exec_fn[std::make_tuple(bm_, bn_, bk_, wm_, wn_, wk_, mm_, mn_, mk_, nstage_)];

    spatha::SpmmBlockwiseOpState state = (*init_func)(spmat, n, d.get_Bmat().device_ptr, d.get_Cmat().device_ptr);
    //warmmup
    for(int warm=0; warm<1; warm++){
	    (*exec_func)(state, stream);
    }
    cudaDeviceSynchronize();

    time = cuTime(times, exec_func, state, stream);

    d.get_Cmat().sync_host();

    this->result = std::vector<T2>();
    for (size_t i = 0; i < m*n; i++){
        this->result.push_back(d.get_Cmat().host_ptr[i]);
    }

    return time;
}

template<>
float Spatha<float,float>::spmm(int times, int bm_, int bn_, int bk_, int wm_, int wn_, int wk_, int mm_, int mn_, int mk_, int nstage_){
    std::cerr << "only precision:half is implemented.\n";
    exit(EXIT_FAILURE);
}

template<>
float Spatha<int8_t, int8_t>::spmm(int times, int bm_, int bn_, int bk_, int wm_, int wn_, int wk_, int mm_, int mn_, int mk_, int nstage_){
    std::cerr << "only precision:half is implemented.\n";
    exit(EXIT_FAILURE);
}

template class Spatha<float,float>;
template class Spatha<half,half>;
template class Spatha<int8_t,int8_t>;