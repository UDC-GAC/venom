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

#include "format_sr_nm.hpp"

template<class T>
Format_sr_nm<T>::Format_sr_nm(int nrow, int ncol, int n, int m, int nnz, float density, unsigned seed) :Format<T>(nrow, ncol, nnz, density, seed){
    this->device_synced = false;
};

template<class T>
Format_sr_nm<T>::~Format_sr_nm(){
    if (device_synced) {
        __clear_device_ref();
    }
};

template<class T>
void Format_sr_nm<T>::init(int nrow_, int ncol_, int n_, float density_, unsigned seed_, bool row_permute_, int m_, int brow_, int mbrow_, int bm_){

    this->A_num_rows = nrow_;
    this->A_num_cols = ncol_;
    this->A_size = nrow_*ncol_;
    this->n = n_;
    this->m = m_;
    this->density = (float)n_/(float)m_;
    this->seed = seed_;
    this->brow = brow_;
    this->mbrow = mbrow_;
    this->bm   = bm_;
    // !IMPORTANT! constants because of architecture constraints
    m_fixed = 4;
    bits_elem_meta=2;
    mrow_m = 2;
    bits_elem_cols=8;
    brow_fixed = 16;
    nelems=(sizeof(uint)*8)/bits_elem_meta;
    nelems_col = nelems/mrow_m;

    A_num_cols_sp = (ncol_/m)*n;
    A_num_cols_sp_pad_nm = (ROUND_UP(this->A_num_cols, m)/m)*n;
    A_num_cols_sp_pad = ROUND_UP((ROUND_UP(ncol_, m)/m)*n, 16); //16=mma_K/2
    this->A_nnz = this->A_num_rows*A_num_cols_sp_pad;

    this->hA_values.resize(this->A_nnz, 0);
    this->hA_columns.resize(this->A_num_rows/bm_ * A_num_cols_sp_pad/n*m_fixed, 0);//4=m_fixed
    this->hA_metadata.resize(this->A_num_rows/mrow_m * A_num_cols_sp_pad/nelems_col, 0);

    Format_sr_nm<T>::random_mask(nrow_, A_num_cols_sp, A_num_cols_sp_pad_nm, n_, m_, mbrow_, this->density, this->hA_metadata, this->hA_columns, seed_, bm_);

    for (auto it = this->hA_values.begin(); it != this->hA_values.end(); it++) {
        *it = static_cast<T>((float)(std::rand() % 9 - 4));
        //*it = static_cast<T>((float)(std::rand() % 8 + 1));
        //*it = static_cast<T>((float)(1));
    }

    this->initialized = true;
    if (this->device_synced) {
        this->__clear_device_ref();
        this->device_synced = false;
    }
}

template<class T>
void Format_sr_nm<T>::init_from_sparse(int nrow_, int ncol_, float density_, unsigned seed_, int n_, int m_, int brow_, int mbrow_, int bm_, std::vector<T>& values_, std::vector<uint>& columns_, std::vector<uint>& metadata_){
    this->A_num_rows = nrow_;
    this->A_num_cols = ncol_;
    this->A_size = nrow_*ncol_;
    this->n = n_;
    this->m = m_;
    this->density = (float)n_/(float)m_;
    this->seed = seed_;
    this->brow = brow_;
    this->mbrow = mbrow_;
    this->bm   = bm_;
    // !IMPORTANT! constants because of architecture constraints
    m_fixed = 4;
    bits_elem_meta=2;
    mrow_m = 2;
    bits_elem_cols=8;
    brow_fixed = 16;
    nelems=(sizeof(uint)*8)/bits_elem_meta;
    nelems_col = nelems/mrow_m;

    A_num_cols_sp = (ncol_/m)*n;
    A_num_cols_sp_pad_nm = (ROUND_UP(this->A_num_cols, m)/m)*n;
    A_num_cols_sp_pad = ROUND_UP((ROUND_UP(ncol_, m)/m)*n, 16); //16=mma_K/2
    this->A_nnz = this->A_num_rows*A_num_cols_sp_pad;

    this->hA_metadata = metadata_;
    this->hA_values   = values_;
    this->hA_columns  = columns_;
}

template<class T>
void Format_sr_nm<T>::init_from_dense(int nrow_, int ncol_, float density_, unsigned seed_, int n_, int m_, int brow_, int mbrow_, int bm_, std::vector<T>& dense, std::vector<uint>& columns_)
{
    this->A_num_rows = nrow_;
    this->A_num_cols = ncol_;
    this->A_size = nrow_*ncol_;
    this->n = n_;
    this->m = m_;
    this->density = (float)n_/(float)m_;
    this->seed = seed_;
    this->brow = brow_;
    this->mbrow = mbrow_;
    this->bm   = bm_;
    // !IMPORTANT! constants because of architecture constraints
    m_fixed = 4;
    bits_elem_meta=2;
    mrow_m = 2;
    bits_elem_cols=8;
    brow_fixed = 16;
    nelems=(sizeof(uint)*8)/bits_elem_meta;
    nelems_col = nelems/mrow_m;

    A_num_cols_sp = (ncol_/m)*n;
    A_num_cols_sp_pad_nm = (ROUND_UP(this->A_num_cols, m)/m)*n;
    A_num_cols_sp_pad = ROUND_UP((ROUND_UP(ncol_, m)/m)*n, 16); //16=mma_K/2
    this->A_nnz = this->A_num_rows*A_num_cols_sp_pad;

    //this->hA_values.resize(this->A_nnz, static_cast<T>(0.0f));
    this->hA_metadata.resize(this->A_num_rows/mrow_m * A_num_cols_sp_pad/nelems_col, 0);
    this->hA_values.resize(this->A_nnz, 0);
    std::cout << "hA_values size " << this->hA_values.size() << std::endl;
    /* for (auto i=0; i<this->A_nnz; i++) {
        this->hA_values.push_back(static_cast<T>(0.0f)); } */
    for (auto it = this->hA_values.begin(); it != this->hA_values.end(); it++) {
        *it = static_cast<T>(0.0f);
    }

    this->hA_columns = columns_;

    std::cout << "to_sparse_sr_nm" << std::endl;
    Format_sr_nm<T>::to_sparse_sr_nm(dense); //FIXME: columns as parameter
}

template<typename T>
void Format_sr_nm<T>::random_mask(int nrow, int ncol, int ncol_pad, int n, int m, int mbrow, float density, std::vector<uint> &metadata, std::vector<uint> &indices, unsigned seed, int bm_){
    std::vector<unsigned int> arr;
    for(int i = 0; i < m_fixed; i++){ arr.push_back(i); }

    // metadata
    for(int i=0; i<nrow/mbrow; i++){
        int j=0;
        for(; j<ncol/nelems_col; j++){
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
                hA_metadata[
                        i*mbrow/mrow_m*A_num_cols_sp_pad/nelems_col+
                        j*mbrow/mrow_m+
                        k] = meta;
            }
        }

        if(ncol_pad>ncol){
            for(int k=0; k<mbrow/mrow_m; k++){
                unsigned int meta=0;
                int resto = (this->A_num_cols)-(4*m*(ncol/nelems_col));
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

                hA_metadata[i*mbrow/2*A_num_cols_sp_pad/8+
                            j*mbrow/2+
                            k] = meta;
            }
        }
    }

    std::vector<unsigned int> arr2;
    for(int i = 0; i < m; i++){ arr2.push_back(i); }

    int resto = this->A_num_cols-(ncol/n)*m;
    for(int i=0; i<(nrow/bm_); i++){
        int j=0;
        for(; j<ncol/n; j++){
            random_shuffle(arr2.begin(), arr2.end());
            std::sort(arr2.begin(), arr2.begin() + m_fixed);

            for(int w=0; w<m_fixed; w++){
                uint tmp = arr2[w];

                this->hA_columns[i*(A_num_cols_sp_pad/n)*m_fixed +
                                 j*m_fixed + w] = tmp;
            }
        }
        if(resto>0){
            for(int w=0; w<m_fixed; w++){
                this->hA_columns[i*(A_num_cols_sp_pad/n)*m_fixed +
                                    j*m_fixed + w] = w;
            }
        }
    }
}

template<typename T>
void Format_sr_nm<T>::transform_and_sync_device() {

    assert(this->initialized && "must call initialize() before sync to device.\n");

    // create host format
    int size_of_meta = sizeof(uint)*(ROUND_UP(this->hA_metadata.size(), this->alignment)+ ROUND_UP(this->hA_columns.size(), this->alignment));
    this->meta_h = std::vector<char>(size_of_meta);
    memset(this->meta_h.data(), 0x0, size_of_meta*sizeof(char));

    int offset = 0;
    memcpy(this->meta_h.data(), this->hA_metadata.data(), sizeof(uint)*this->hA_metadata.size());
    offset+=this->hA_metadata.size();
    while(offset&this->alignment != 0) offset++;

    memcpy(this->meta_h.data() + offset * (sizeof(uint)/sizeof(char)), this->hA_columns.data(), sizeof(uint)*this->hA_columns.size());

    this->data_h = this->hA_values;

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

    offset += this->hA_metadata.size();
    while(offset % this->alignment != 0) offset++;
    this->device_ref.csb_indices = (uint*) meta_d + offset;

    this->device_ref.opdA = data_d;

    // set flag
    device_synced = true;
}

template<class T>
std::vector<unsigned int>& Format_sr_nm<T>::get_hA_metadata(){
    return this->hA_metadata;
}

template<class T>
std::vector<unsigned int>& Format_sr_nm<T>::get_hA_indices(){
    return this->hA_columns;
}

template<class T>
std::vector<T>& Format_sr_nm<T>::get_hA_val(){
    return this->hA_values;
}

template<class T>
int Format_sr_nm<T>::get_n(){
    return this->n;
}

template<class T>
int Format_sr_nm<T>::get_vec_length(){
    return this->brow;
}

template<class T>
int Format_sr_nm<T>::get_meta_vec_length(){
    return this->mbrow;
}

template<class T>
int Format_sr_nm<T>::get_m(){
    return this->m;
}

template<class T>
int Format_sr_nm<T>::get_bm(){
    return this->bm;
}

template<class T>
std::vector<T>& Format_sr_nm<T>::to_dense(){
    /* for (size_t i = 0; i < this->A_size; i++) { this->hA_dense.push_back(0.0f); } */
    this->hA_dense.resize(this->A_size, 0);

    // general variables N:M format
    int bm_m = this->A_num_rows/this->bm;
    int mbrow_m = bm/this->mbrow;
    int mbrow_m2 = this->mbrow/brow_fixed;
    int brow_m = brow_fixed/this->brow;
    // metadata
    int mcol_kk = nelems/mrow_m/n;
    int mcol_k = A_num_cols_sp_pad/n/mcol_kk;
    // indices
    int col_kk = mcol_kk;
    int col_k = A_num_cols_sp_pad/n/col_kk;

    uint indexes[nelems];
    uint columns[col_kk*m_fixed];

    for(int bm_i=0; bm_i<bm_m; bm_i++){ // tileM_i
        for(int mbrow_i=0; mbrow_i<mbrow_m; mbrow_i++){ // metadata_i
            for(int mbrow_i2=0; mbrow_i2<mbrow_m2; mbrow_i2++){ // metadata_i
                for(int brow_i=0; brow_i<brow_m; brow_i++){ // thread {0,4,8,...}
                    for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){ // tileK_i
                        //read columns indexes
                        for(int col_i=0; col_i<col_kk; col_i++){
                            for(int col_ii=0; col_ii<m_fixed; col_ii++){
                                columns[col_i*m_fixed + col_ii] =
                                hA_columns[bm_i*col_k*col_kk*m_fixed + mcol_i*col_kk*m_fixed + col_i*m_fixed + col_ii];
                            }
                        }
                        // read metadata
                        for(int mbrow_ii=0; mbrow_ii<(brow/mrow_m); mbrow_ii++){
                            for(int mbrow_iii=0; mbrow_iii<mrow_m; mbrow_iii++){
                                for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){
                                    for (int n_i=0; n_i<n; n_i++) {
                                        indexes[
                                            mbrow_iii*n +
                                            mcol_ii*mrow_m*n +
                                            n_i] =
                                        (((hA_metadata[
                                            bm_i*mcol_k*bm/mrow_m +
                                            mbrow_i*mcol_k*mbrow/mrow_m +
                                            mbrow_i2*brow_fixed/mrow_m +
                                            brow_i*brow/mrow_m  +
                                            mcol_i*mbrow/mrow_m +
                                            mbrow_ii]) >> (mbrow_iii*(nelems/mrow_m)*bits_elem_meta+mcol_ii*n*bits_elem_meta+n_i*bits_elem_meta)) & 0x3);
                                    }
                                }
                            }

                            for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){
                                for(int mbrow_iii=0; mbrow_iii<mrow_m; mbrow_iii++){
                                    for(int n_i=0; n_i<n; n_i++){
                                        unsigned int index = columns[mcol_ii*m_fixed + indexes[mcol_ii*mrow_m*n+mbrow_iii*n+n_i]];

                                        if((mcol_i*m*mcol_kk + mcol_ii*m + index) < this->A_num_cols){
                                            this->hA_dense[
                                                bm_i*bm*this->A_num_cols +
                                                mbrow_i*mbrow*this->A_num_cols +
                                                mbrow_i2*brow_fixed*this->A_num_cols +
                                                brow_i*brow*this->A_num_cols +
                                                mcol_i*m*mcol_kk +
                                                mbrow_ii*mrow_m*this->A_num_cols +
                                                mcol_ii*m +
                                                mbrow_iii*this->A_num_cols +
                                                index] =
                                            this->hA_values[
                                                bm_i*bm*A_num_cols_sp_pad +
                                                mbrow_i*mbrow*A_num_cols_sp_pad+
                                                mbrow_i2*brow_fixed*A_num_cols_sp_pad+
                                                brow_i*brow*nelems/mrow_m+
                                                mcol_i*brow_fixed*nelems/mrow_m +
                                                mbrow_ii*mrow_m*n +
                                                mcol_ii*n*brow +
                                                mbrow_iii*n +
                                                n_i];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return this->hA_dense;
}

template<class T>
void Format_sr_nm<T>::to_sparse_sr_nm(std::vector<T>& dense){
    //for (size_t i = 0; i < this->A_size; i++) { this->hA_dense.push_back(0.0f); }
    //this->hA_values.resize(this->A_nnz, 0);
    //this->hA_metadata.resize(this->A_num_rows/mrow_m * A_num_cols_sp_pad/nelems_col, 0);

    // general variables N:M format
    int bm_m = this->A_num_rows/this->bm;
    int mbrow_m = bm/this->mbrow;
    int mbrow_m2 = this->mbrow/brow_fixed;
    int brow_m = brow_fixed/this->brow;
    // metadata
    int mcol_kk = nelems/mrow_m/n;
    int mcol_k = A_num_cols_sp_pad/n/mcol_kk;
    // indices
    int col_kk = mcol_kk;
    int col_k = A_num_cols_sp_pad/n/col_kk;

    T values[nelems];
    uint indexes[nelems];
    uint columns[col_kk*m_fixed];

    for(int bm_i=0; bm_i<bm_m; bm_i++){ // tileM_i
        for(int mbrow_i=0; mbrow_i<mbrow_m; mbrow_i++){ // metadata_i
            for(int mbrow_i2=0; mbrow_i2<mbrow_m2; mbrow_i2++){ // metadata_i
                for(int brow_i=0; brow_i<brow_m; brow_i++){ // thread {0,4,8,...}
                    for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){ // tileK_i
                        //read columns indexes
                        for(int col_i=0; col_i<col_kk; col_i++){
                            for(int col_ii=0; col_ii<m_fixed; col_ii++){
                                columns[col_i*m_fixed + col_ii] =
                                hA_columns[bm_i*col_k*col_kk*m_fixed + mcol_i*col_kk*m_fixed + col_i*m_fixed + col_ii];
                            }
                        }
                        for(int mbrow_ii=0; mbrow_ii<(brow/mrow_m); mbrow_ii++){
                            // read dense matrix
                            for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){
                                for(int mbrow_iii=0; mbrow_iii<mrow_m; mbrow_iii++){
                                    //for(int n_i=0; n_i<n; n_i++){
                                    int pos=0;
                                    for(int n_i=0; n_i<m_fixed; n_i++){
                                        //unsigned int index = columns[mcol_ii*m_fixed + indexes[mcol_ii*mrow_m*n+mbrow_iii*n+n_i]];
                                        unsigned int index = columns[mcol_ii*m_fixed + n_i];

                                        if((mcol_i*m*mcol_kk + mcol_ii*m + index) < this->A_num_cols){
                                            T val = dense[
                                                    bm_i*bm*this->A_num_cols +
                                                    mbrow_i*mbrow*this->A_num_cols +
                                                    mbrow_i2*brow_fixed*this->A_num_cols +
                                                    brow_i*brow*this->A_num_cols +
                                                    mcol_i*m*mcol_kk +
                                                    mbrow_ii*mrow_m*this->A_num_cols +
                                                    mcol_ii*m +
                                                    mbrow_iii*this->A_num_cols +
                                                    index];
                                            if(static_cast<float>(val) != 0){ //FIXME: check with mask, not value
                                                indexes[
                                                    mbrow_iii*n +
                                                    mcol_ii*mrow_m*n +
                                                    pos] = n_i;

                                                values[
                                                    mcol_ii*mrow_m*n +
                                                    mbrow_iii*n +
                                                    pos] = val;
                                                pos+=1;
                                            }
                                        } else {
                                            if(n_i<2){
                                                indexes[
                                                    mbrow_iii*n +
                                                    mcol_ii*mrow_m*n +
                                                    pos] = 0;

                                                values[
                                                    mcol_ii*mrow_m*n +
                                                    mbrow_iii*n +
                                                    pos] = 0;

                                                pos+=1;
                                            }
                                        }
                                    }
                                }
                            }
                            // write metadata
                            unsigned int meta=0;
                            for(int mbrow_iii=0; mbrow_iii<mrow_m; mbrow_iii++){
                                for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){
                                    for (int n_i=0; n_i<n; n_i++) {
                                        this->hA_values[
                                                bm_i*bm*A_num_cols_sp_pad +
                                                mbrow_i*mbrow*A_num_cols_sp_pad+
                                                mbrow_i2*brow_fixed*A_num_cols_sp_pad+
                                                brow_i*brow*nelems/mrow_m+
                                                mcol_i*brow_fixed*nelems/mrow_m +
                                                mbrow_ii*mrow_m*n +
                                                mcol_ii*n*brow +
                                                mbrow_iii*n +
                                                n_i] =
                                        values[mcol_ii*mrow_m*n +
                                                mbrow_iii*n +
                                                n_i];

                                        unsigned int tmp = indexes[
                                                    mbrow_iii*n +
                                                    mcol_ii*mrow_m*n +
                                                    n_i];
                                        meta |= (tmp << (mbrow_iii*(nelems/mrow_m)*bits_elem_meta+mcol_ii*n*bits_elem_meta+n_i*bits_elem_meta));
                                    }
                                }
                            }
                            hA_metadata[bm_i*mcol_k*bm/mrow_m +
                                        mbrow_i*mcol_k*mbrow/mrow_m +
                                        mbrow_i2*brow_fixed/mrow_m +
                                        brow_i*brow/mrow_m  +
                                        mcol_i*mbrow/mrow_m +
                                        mbrow_ii] = meta;
                        }
                    }
                }
            }
        }
    }
}

template<class T>
typename Format_sr_nm<T>::DeviceRef& Format_sr_nm<T>::get_device_ref(){
    return this->device_ref;
}

template<class T>
void Format_sr_nm<T>::__clear_device_ref() {
    if (device_ref._meta)
        CUDA_CHECK( cudaFree(device_ref._meta));
    if (device_ref._data)
        CUDA_CHECK( cudaFree(device_ref._data));
}

template class Format_sr_nm<float>;
template class Format_sr_nm<__half>;
template class Format_sr_nm<int8_t>;