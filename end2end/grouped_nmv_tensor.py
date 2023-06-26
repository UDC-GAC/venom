#
# Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from native_scripting import compile
import functools
import ctypes
from heapq import nlargest
import math

try:
    cache = functools.cache
except AttributeError:
    cache = functools.lru_cache(maxsize=None)

class FixedMaskTensor:
    def __init__(self, val, mask, n, m, tileM):
        assert torch.all(
            torch.isclose(mask, torch.zeros_like(mask))
            | torch.isclose(mask, torch.ones_like(mask))
        )
        self.val = val
        self.mask = mask
        self.n = n
        self.m = m
        self.tileM = tileM

    @staticmethod
    def from_dense(tensor, n, m, tileM):
        mask = torch.where(
            tensor.abs() < 1e-6,
            torch.zeros_like(tensor, dtype=torch.bool),
            torch.ones_like(tensor, dtype=torch.bool),
        )
        return FixedMaskTensor(tensor * mask, mask, n, m, tileM)

    def to_dense(self):
        return copy.deepcopy(self.val)

    def numel(self):
        return self.val.numel()

    def to(self, device=None, dtype=None, non_blocking=False, copy=False):
        return FixedMaskTensor(
            self.val.to(device=device, dtype=dtype, copy=True),
            self.mask.to(device=device, dtype=dtype, copy=True),
            self.n,
            self.m,
            self.tileM,
        )

    @property
    def shape(self):
        return self.val.shape

    @property
    def device(self):
        return self.val.device

    @property
    def dtype(self):
        return self.val.dtype


def round_up(x,y):
    return math.ceil(x/y)*y

@cache
def group_n_m2(dense_shape, dense_dtype, n, m, tileM):
    nrows = dense_shape[0]
    ncols = dense_shape[1]

    m_fixed = 4
    A_num_cols_sp_pad = round_up((round_up(ncols, m)/m)*n, 16)
    indexes_cols = A_num_cols_sp_pad//n*m_fixed

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    lib = compile(
        f"""
        #include <iostream>
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <numeric>
        #include <chrono>

        using namespace std;

        int int_ceil(int x, int y){{
            return (x - 1) / y + 1;
        }}

        extern "C" void func({dtype}* dense, {dtype}* sparse, int* masks, int *columns){{
            int bm_m   = {nrows}/{tileM};
            int mcol_k = {ncols}/{m};
            int mcol_k_p = int_ceil({ncols},{m});
            int m_fixed = 4;

            std::vector<{dtype}> v(m_fixed, 0);
            std::vector<int> vx(m_fixed, 0);
            {dtype} max=0, total=0;

            std::vector<size_t> indices(v.size());
            std::iota(indices.begin(), indices.end(), 0);

            for(int bm_i=0; bm_i<bm_m; bm_i++){{
                int t_bm_i   = bm_i*{tileM}*{ncols};
                for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){{
                    int t_mcol_i = mcol_i*{m};
                    max = 0;

                    std::vector<int> cols_max;
                    cols_max.resize(m_fixed, 0);
                    std::vector<int> masks_max;
                    masks_max.resize({tileM}*{m}, 0);

                    for(int col_i=0; col_i<{m}; col_i++){{
                        vx[0]=col_i;
                        for(int col_j=col_i+1; col_j<{m}; col_j++){{
                            vx[1]=col_j;
                            for(int col_k=col_j+1; col_k<{m}; col_k++){{
                                vx[2]=col_k;
                                for(int col_w=col_k+1; col_w<{m}; col_w++){{
                                    vx[3]=col_w;
                                    total=0;

                                    std::vector<int> mask({tileM}*{m}, 0);

                                    for(int row_i=0; row_i<{tileM}; row_i++){{
                                        int t_row_i  = row_i*{ncols};
                                        v[0]=dense[t_bm_i + t_row_i + t_mcol_i + col_i];
                                        v[1]=dense[t_bm_i + t_row_i + t_mcol_i + col_j];
                                        v[2]=dense[t_bm_i + t_row_i + t_mcol_i + col_k];
                                        v[3]=dense[t_bm_i + t_row_i + t_mcol_i + col_w];

                                        std::partial_sort(indices.begin(), indices.begin() + {n}, indices.end(), [&](size_t A, size_t B) {{
                                                    return v[A] > v[B];}});

                                        for(int id=0; id<{n}; id++){{
                                            total += dense[t_bm_i + t_row_i + t_mcol_i + vx[indices[id]]];

                                            mask[row_i*{m} + vx[indices[id]]] = 1;
                                        }}
                                    }}

                                    if(total>max){{
                                        max = total;
                                        std::copy(mask.begin(), mask.end(), masks_max.begin());

                                        std::sort(vx.begin(), vx.begin() + m_fixed);
                                        std::copy(vx.begin(), vx.end(), cols_max.begin());
                                    }}
                                }}
                            }}
                        }}
                    }}

                    for(int i=0; i<{tileM}; i++){{
                        for(int j=0; j<{m}; j++){{
                            int drop = masks_max[i*{m} + j];
                            masks[t_bm_i  + i*{ncols}+ t_mcol_i + j]  = drop;
                            sparse[t_bm_i + i*{ncols}+ t_mcol_i + j] *= drop;
                        }}
                    }}
                    for(int i=0; i<m_fixed; i++){{
                        columns[bm_i*{indexes_cols} + mcol_i*m_fixed + i] =
                        cols_max[i];
                    }}
                }}
            }}

            int remainder = {ncols}%{m};

            if (remainder>0){{
                int d_rows={tileM}, d_cols=remainder;

                if(remainder<3){{
                    for(int i=0; i<{nrows}; i++){{
                        for(int j={ncols}-remainder; j<{ncols}; j++){{
                            masks[i*{ncols}+j] = 1;
                        }}
                    }}
                    for(int bm_i=0; bm_i<bm_m; bm_i++){{
                        for(int j=0; j<m_fixed; j++){{
                            columns[bm_i*{indexes_cols} + mcol_k*m_fixed + j] = j;
                        }}
                    }}
                }} else if(remainder==3){{
                    v[3] = -1;
                    for(int bm_i=0; bm_i<bm_m; bm_i++){{
                        int t_bm_i   = bm_i*{tileM}*{ncols};
                        for(int mcol_i=mcol_k; mcol_i<mcol_k_p; mcol_i++){{
                            max = 0;
                            int t_mcol_i = mcol_i*{m};

                            std::vector<int> cols_max(m_fixed, 0);
                            std::vector<int> masks_max({tileM}*remainder, 0);

                            for(int col_i=0; col_i<remainder; col_i++){{
                                vx[0]=col_i;
                                for(int col_j=col_i+1; col_j<remainder; col_j++){{
                                    vx[1]=col_j;
                                    for(int col_k=col_j+1; col_k<remainder; col_k++){{
                                        vx[2]=col_k;
                                        total=0;
                                        std::vector<int> mask({tileM}*remainder, 0);

                                        for(int row_i=0; row_i<{tileM}; row_i++){{
                                            int t_row_i  = row_i*{ncols};
                                            v[0]=dense[t_bm_i + t_row_i + t_mcol_i + col_i];
                                            v[1]=dense[t_bm_i + t_row_i + t_mcol_i + col_j];
                                            v[2]=dense[t_bm_i + t_row_i + t_mcol_i + col_k];

                                            std::partial_sort(indices.begin(), indices.begin() + {n}, indices.end(), [&](size_t A, size_t B) {{
                                                        return v[A] > v[B]; }});

                                            for(int id=0; id<{n}; id++){{
                                                total += dense[t_bm_i + t_row_i + t_mcol_i + vx[indices[id]]];

                                                mask[row_i*remainder + vx[indices[id]]] = 1;
                                            }}
                                        }}

                                        if(total>max){{
                                            max = total;
                                            std::copy(mask.begin(), mask.end(), masks_max.begin());

                                            std::sort(vx.begin(), vx.begin() + remainder);//m_fixed
                                            std::copy(vx.begin(), vx.end(), cols_max.begin());
                                        }}
                                    }}
                                }}
                            }}

                            for(int i=0; i<{tileM}; i++){{
                                for(int j=0; j<remainder; j++){{
                                    int drop = masks_max[i*remainder + j];

                                    masks[t_bm_i  + i*{ncols}+ t_mcol_i + j]  = drop;
                                    sparse[t_bm_i + i*{ncols}+ t_mcol_i + j] *= drop;
                                }}
                            }}
                            for(int i=0; i<remainder; i++){{
                                columns[bm_i*{indexes_cols} + mcol_i*m_fixed + i] =
                                cols_max[i];
                            }}
                        }}
                    }}
                }} else {{
                    for(int bm_i=0; bm_i<bm_m; bm_i++){{
                        int t_bm_i   = bm_i*{tileM}*{ncols};
                        for(int mcol_i=mcol_k; mcol_i<mcol_k_p; mcol_i++){{
                            max = 0;
                            int t_mcol_i = mcol_i*{m};

                            std::vector<int> cols_max(m_fixed, 0);
                            std::vector<int> masks_max({tileM}*remainder, 0);

                            for(int col_i=0; col_i<remainder; col_i++){{
                                vx[0]=col_i;
                                for(int col_j=col_i+1; col_j<remainder; col_j++){{
                                    vx[1]=col_j;
                                    for(int col_k=col_j+1; col_k<remainder; col_k++){{
                                        vx[2]=col_k;
                                        for(int col_w=col_k+1; col_w<remainder; col_w++){{
                                            vx[3]=col_w;
                                            total=0;
                                            std::vector<int> mask({tileM}*remainder, 0);

                                            for(int row_i=0; row_i<{tileM}; row_i++){{
                                                int t_row_i  = row_i*{ncols};
                                                v[0]=dense[t_bm_i + t_row_i + t_mcol_i + col_i];
                                                v[1]=dense[t_bm_i + t_row_i + t_mcol_i + col_j];
                                                v[2]=dense[t_bm_i + t_row_i + t_mcol_i + col_k];
                                                v[3]=dense[t_bm_i + t_row_i + t_mcol_i + col_w];

                                                std::partial_sort(indices.begin(), indices.begin() + {n}, indices.end(), [&](size_t A, size_t B) {{
                                                            return v[A] > v[B]; }});

                                                for(int id=0; id<{n}; id++){{
                                                    total += dense[t_bm_i + t_row_i + t_mcol_i + vx[indices[id]]];

                                                    mask[row_i*remainder + vx[indices[id]]] = 1;
                                                }}
                                            }}

                                            if(total>max){{
                                                max = total;
                                                std::copy(mask.begin(), mask.end(), masks_max.begin());

                                                std::sort(vx.begin(), vx.begin() + m_fixed);
                                                std::copy(vx.begin(), vx.end(), cols_max.begin());
                                            }}
                                        }}
                                    }}
                                }}
                            }}

                            for(int i=0; i<{tileM}; i++){{
                                for(int j=0; j<remainder; j++){{
                                    int drop = masks_max[i*remainder + j];

                                    masks[t_bm_i  + i*{ncols}+ t_mcol_i + j]  = drop;
                                    sparse[t_bm_i + i*{ncols}+ t_mcol_i + j] *= drop;
                                }}
                            }}
                            for(int i=0; i<m_fixed; i++){{
                                columns[bm_i*{indexes_cols} + mcol_i*m_fixed + i] =
                                cols_max[i];
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """,
    )
    lib.func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib.func

@cache
def to_dense(dense_shape, dense_dtype, n, m, tileM):
    nrows = dense_shape[0]
    ncols = dense_shape[1]

    A_size = nrows*ncols
    density = n/m

    brow = 4 #this->brow = brow_;
    mbrow = 32 #this->mbrow = mbrow_;

    bm   = tileM
    # !IMPORTANT! constants because of architecture constraints
    m_fixed = 4
    bits_elem_meta=2
    mrow_m = 2
    bits_elem_cols=8
    brow_fixed = 16
    nelems=32//bits_elem_meta #(sizeof(uint)*8)=32
    nelems_col = nelems//mrow_m

    A_num_cols_sp = (ncols/m)*n
    A_num_cols_sp_pad_nm = (round_up(ncols, m)/m)*n
    A_num_cols_sp_pad = round_up((round_up(ncols, m)/m)*n, 16)
    A_nnz = nrows*A_num_cols_sp_pad

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    lib = compile(
        f"""
        #include <iostream>
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <numeric>
        #include <chrono>

        using namespace std;


        extern "C" void func3({dtype}* hA_dense, {dtype}* hA_values, int *hA_columns, int *hA_metadata){{
            //this->hA_dense.resize(this->A_size, 0);

            // general variables N:M format
            int bm_m = {nrows}/{bm};
            int mbrow_m = {bm}/{mbrow};
            int mbrow_m2 = {mbrow}/{brow_fixed};
            int brow_m = {brow_fixed}/{brow};
            // metadata
            int mcol_kk = {nelems}/{mrow_m}/{n};
            int mcol_k = {A_num_cols_sp_pad}/{n}/mcol_kk;
            // indices
            int col_kk = mcol_kk;
            int col_k = {A_num_cols_sp_pad}/{n}/col_kk;

            uint indexes[{nelems}];
            uint columns[col_kk*{m_fixed}];

            for(int bm_i=0; bm_i<bm_m; bm_i++){{
                for(int mbrow_i=0; mbrow_i<mbrow_m; mbrow_i++){{
                    for(int mbrow_i2=0; mbrow_i2<mbrow_m2; mbrow_i2++){{
                        for(int brow_i=0; brow_i<brow_m; brow_i++){{
                            for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){{
                                //read columns indexes
                                for(int col_i=0; col_i<col_kk; col_i++){{
                                    for(int col_ii=0; col_ii<{m_fixed}; col_ii++){{
                                        columns[col_i*{m_fixed} + col_ii] =
                                        hA_columns[bm_i*col_k*col_kk*{m_fixed} + mcol_i*col_kk*{m_fixed} + col_i*{m_fixed} + col_ii];
                                    }}
                                }}
                                // read metadata
                                for(int mbrow_ii=0; mbrow_ii<({brow}/{mrow_m}); mbrow_ii++){{
                                    for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                        for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                            for (int n_i=0; n_i<{n}; n_i++) {{
                                                indexes[
                                                    mbrow_iii*{n} +
                                                    mcol_ii*{mrow_m}*{n} +
                                                    n_i] =
                                                (((hA_metadata[
                                                    bm_i*mcol_k*{bm}/{mrow_m} +
                                                    mbrow_i*mcol_k*{mbrow}/{mrow_m} +
                                                    mbrow_i2*{brow_fixed}/{mrow_m} +
                                                    brow_i*{brow}/{mrow_m}  +
                                                    mcol_i*{mbrow}/{mrow_m} +
                                                    mbrow_ii]) >> (mbrow_iii*({nelems}/{mrow_m})*{bits_elem_meta}+mcol_ii*{n}*{bits_elem_meta}+n_i*{bits_elem_meta})) & 0x3);
                                            }}
                                        }}
                                    }}

                                    for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                        for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                            for(int n_i=0; n_i<{n}; n_i++){{
                                                unsigned int index = columns[mcol_ii*{m_fixed} + indexes[mcol_ii*{mrow_m}*{n}+mbrow_iii*{n}+n_i]];

                                                if((mcol_i*{m}*mcol_kk + mcol_ii*{m} + index) < {ncols}){{
                                                    hA_dense[
                                                        bm_i*{bm}*{ncols} +
                                                        mbrow_i*{mbrow}*{ncols} +
                                                        mbrow_i2*{brow_fixed}*{ncols} +
                                                        brow_i*{brow}*{ncols} +
                                                        mcol_i*{m}*mcol_kk +
                                                        mbrow_ii*{mrow_m}*{ncols} +
                                                        mcol_ii*{m} +
                                                        mbrow_iii*{ncols} +
                                                        index] =
                                                    hA_values[
                                                        bm_i*{bm}*{A_num_cols_sp_pad} +
                                                        mbrow_i*{mbrow}*{A_num_cols_sp_pad}+
                                                        mbrow_i2*{brow_fixed}*{A_num_cols_sp_pad}+
                                                        brow_i*{brow}*{nelems}/{mrow_m}+
                                                        mcol_i*{brow_fixed}*{nelems}/{mrow_m} +
                                                        mbrow_ii*{mrow_m}*{n} +
                                                        mcol_ii*{n}*{brow} +
                                                        mbrow_iii*{n} +
                                                        n_i];
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """,
    )
    lib.func3.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib.func3

@cache
def to_sparse_sr_nm(dense_shape, dense_dtype, n, m, tileM):
    nrows = dense_shape[0]
    ncols = dense_shape[1]

    brow = 4 #this->brow = brow_;
    mbrow = 32 #this->mbrow = mbrow_;

    bm   = tileM
    # !IMPORTANT! constants because of architecture constraints
    m_fixed = 4
    bits_elem_meta=2
    mrow_m = 2
    bits_elem_cols=8
    brow_fixed = 16
    nelems=32//bits_elem_meta #(sizeof(uint)*8)=32
    nelems_col = nelems//mrow_m

    A_num_cols_sp = (ncols//m)*n
    A_num_cols_sp_pad_nm = (round_up(ncols, m)/m)*n
    A_num_cols_sp_pad = round_up((round_up(ncols, m)/m)*n, 16)
    A_nnz = nrows*A_num_cols_sp_pad

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    lib = compile(
        f"""
        #include <iostream>
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <numeric>
        #include <chrono>

        using namespace std;


        extern "C" void func2({dtype}* sparse, int* masks, {dtype}* hA_values, int *hA_columns, int *hA_metadata){{

            int bm_m = {nrows}/{bm};
            int mbrow_m = {bm}/{mbrow};
            int mbrow_m2 = {mbrow}/{brow_fixed};
            int brow_m = {brow_fixed}/{brow};
            // metadata
            int mcol_kk = {nelems}/{mrow_m}/{n};
            int mcol_k = {A_num_cols_sp_pad}/{n}/mcol_kk;
            // indices
            int col_kk = mcol_kk;
            int col_k = {A_num_cols_sp_pad}/{n}/col_kk;

            {dtype} values[{nelems}];
            uint indexes[{nelems}];
            uint columns[col_kk*{m_fixed}];

            int max_idx = 0;

            for(int bm_i=0; bm_i<bm_m; bm_i++){{
                for(int mbrow_i=0; mbrow_i<mbrow_m; mbrow_i++){{
                    for(int mbrow_i2=0; mbrow_i2<mbrow_m2; mbrow_i2++){{
                        for(int brow_i=0; brow_i<brow_m; brow_i++){{
                            for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){{
                                for(int col_i=0; col_i<col_kk; col_i++){{
                                    for(int col_ii=0; col_ii<{m_fixed}; col_ii++){{
                                        columns[col_i*{m_fixed} + col_ii] =
                                        hA_columns[bm_i*col_k*col_kk*{m_fixed} + mcol_i*col_kk*{m_fixed} + col_i*{m_fixed} + col_ii];
                                    }}
                                }}
                                for(int mbrow_ii=0; mbrow_ii<({brow}/{mrow_m}); mbrow_ii++){{
                                    for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                        for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                            int pos=0;
                                            for(int n_i=0; n_i<{m_fixed}; n_i++){{
                                                unsigned int index = columns[mcol_ii*{m_fixed} + n_i];

                                                if((mcol_i*{m}*mcol_kk + mcol_ii*{m} + index) < {ncols}){{
                                                    int nnz = masks[
                                                            bm_i*{bm}*{ncols} +
                                                            mbrow_i*{mbrow}*{ncols} +
                                                            mbrow_i2*{brow_fixed}*{ncols} +
                                                            brow_i*{brow}*{ncols} +
                                                            mcol_i*{m}*mcol_kk +
                                                            mbrow_ii*{mrow_m}*{ncols} +
                                                            mcol_ii*{m} +
                                                            mbrow_iii*{ncols} +
                                                            index];

                                                    if(nnz != 0){{
                                                        indexes[
                                                            mbrow_iii*{n} +
                                                            mcol_ii*{mrow_m}*{n} +
                                                            pos] = n_i;

                                                        values[
                                                            mcol_ii*{mrow_m}*{n} +
                                                            mbrow_iii*{n} +
                                                            pos] =
                                                        sparse[
                                                            bm_i*{bm}*{ncols} +
                                                            mbrow_i*{mbrow}*{ncols} +
                                                            mbrow_i2*{brow_fixed}*{ncols} +
                                                            brow_i*{brow}*{ncols} +
                                                            mcol_i*{m}*mcol_kk +
                                                            mbrow_ii*{mrow_m}*{ncols} +
                                                            mcol_ii*{m} +
                                                            mbrow_iii*{ncols} +
                                                            index];

                                                        pos+=1;
                                                    }}
                                                }} else {{
                                                    if(n_i<2){{
                                                        indexes[
                                                            mbrow_iii*{n} +
                                                            mcol_ii*{mrow_m}*{n} +
                                                            pos] = 0;

                                                        values[
                                                            mcol_ii*{mrow_m}*{n} +
                                                            mbrow_iii*{n} +
                                                            pos] = 0;

                                                        pos+=1;
                                                    }}
                                                }}
                                            }}
                                        }}
                                    }}
                                    // write metadata
                                    unsigned int meta=0;
                                    for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                        for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                            for (int n_i=0; n_i<{n}; n_i++) {{

                                                int idx = bm_i*{bm}*{A_num_cols_sp_pad} +
                                                        mbrow_i*{mbrow}*{A_num_cols_sp_pad}+
                                                        mbrow_i2*{brow_fixed}*{A_num_cols_sp_pad}+
                                                        brow_i*{brow}*{nelems}/{mrow_m}+
                                                        mcol_i*{brow_fixed}*{nelems}/{mrow_m} +
                                                        mbrow_ii*{mrow_m}*{n} +
                                                        mcol_ii*{n}*{brow} +
                                                        mbrow_iii*{n} +
                                                        n_i;

                                                max_idx = (idx>max_idx)?(idx):(max_idx);

                                                hA_values[
                                                        idx] =
                                                values[
                                                    mcol_ii*{mrow_m}*{n} +
                                                    mbrow_iii*{n} +
                                                    n_i];

                                                unsigned int tmp = indexes[
                                                            mbrow_iii*{n} +
                                                            mcol_ii*{mrow_m}*{n} +
                                                            n_i];
                                                meta |= (tmp << (mbrow_iii*({nelems}/{mrow_m})*{bits_elem_meta}+mcol_ii*{n}*{bits_elem_meta}+n_i*{bits_elem_meta}));
                                            }}
                                        }}
                                    }}
                                    hA_metadata[bm_i*mcol_k*{bm}/{mrow_m} +
                                                mbrow_i*mcol_k*{mbrow}/{mrow_m} +
                                                mbrow_i2*{brow_fixed}/{mrow_m} +
                                                brow_i*{brow}/{mrow_m}  +
                                                mcol_i*{mbrow}/{mrow_m} +
                                                mbrow_ii] = meta;
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            //cout << "max_idx: " << max_idx << endl;
        }}
        """,
    )
    lib.func2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib.func2

class SrNMTensor:
    def __init__(self, n_, m_, tileM_, dense_, mask_, columns_, device_):
        self.n = n_
        self.m = m_
        self.tileM = tileM_
        self.nnz = 0
        self.nrows = None
        self.ncols = None
        self.dense = dense_
        self.device=device_
        self.mask = mask_
        self.columns = columns_
        self.values = None
        self.metadata = None

        self.to_sparse_sr_nm(dense_.cpu().to(dtype=torch.float32), mask_)
        #self.to_sparse_sr_nm(dense_.to(dtype=torch.float32), mask_)
        #self.dense = None

    def to_sparse_sr_nm(self, dense_, mask_):
        impl_builder = (
            to_sparse_sr_nm
            )
        func = impl_builder(
                dense_.shape,
                dense_.dtype,
                self.n,
                self.m,
                self.tileM
            )

        self.nrows, self.ncols = dense_.shape
        A_num_cols_sp_pad = round_up((round_up(self.ncols, self.m)/self.m)*self.n, 16)
        self.nnz = self.nrows*A_num_cols_sp_pad
        m_fixed = 4
        mrow_m = 2
        bits_elem_meta=2

        nelems = 32//bits_elem_meta #32=(sizeof(uint)*8)
        nelems_col = nelems//mrow_m

        self.values = torch.zeros(self.nrows * A_num_cols_sp_pad, dtype=torch.float32) #dense_.dtype
        self.metadata = torch.zeros(self.nrows//mrow_m * A_num_cols_sp_pad//nelems_col, dtype=torch.int32)

        func(dense_.data_ptr(), mask_.data_ptr(), self.values.data_ptr(), self.columns.data_ptr(), self.metadata.data_ptr())

        self.columns  = self.columns.to(device=self.device)
        self.metadata = self.metadata.to(device=self.device)
        self.values   = self.values.to(device=self.device).half()

    def to_dense(self):
        impl_builder = (
            to_dense
            )
        func = impl_builder(
                (self.nrows, self.ncols),
                torch.float32, #self.values.dtype,
                self.n,
                self.m,
                self.tileM
            )
        # initialize with ones
        dense = torch.ones((self.nrows, self.ncols), dtype=torch.float32, device='cpu') #self.values.dtype

        # uncomment to keep initial values
        #func(dense.data_ptr(), self.values.cpu().to(dtype=torch.float32).data_ptr(), self.columns.cpu().data_ptr(), self.metadata.cpu().data_ptr())

        return dense.to(device=self.device).half()

def nm_vector_mask_sparsify(tensor, n, m, tileM):
    #print("nm_vector_mask_sparsify", n, m, tileM)

    impl_builder = (
                group_n_m2
                )
    nrows, ncols = tensor.shape
    A_num_cols_sp_pad = round_up((round_up(ncols, m)/m)*n, 16)
    bm_m   = nrows//tileM
    mcol_k_p = math.ceil(ncols/m)
    m_fixed = 4

    # Structures represent sparse data
    masks = torch.zeros(tensor.shape, dtype=torch.int32, device='cpu')
    columns = torch.zeros(nrows//tileM * A_num_cols_sp_pad//n*m_fixed,dtype=torch.int32, device='cpu')
    if len(tensor.shape) == 2:
        tensor_temp = tensor.to(dtype=torch.float32).cpu().detach().abs()
        sparse = tensor_temp.clone()

        func = impl_builder(
                    tensor_temp.shape,
                    tensor_temp.dtype,
                    n,
                    m,
                    tileM
                )
        func(tensor_temp.data_ptr(), sparse.data_ptr(), masks.data_ptr(), columns.data_ptr())

    else:
        raise NotImplementedError("Only support layers of dimension 2 or 4")

    #self.columns = columns
    #return masks
    return masks, columns