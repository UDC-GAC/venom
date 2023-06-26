#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.ma as ma
import torch
from heapq import nlargest

from native_scripting import compile
import functools
import ctypes

model = torch.hub.load(
    "huggingface/pytorch-transformers", "model", "bert-base-uncased"
)
tensor_name = "encoder.layer.8.attention.self.query.weight"
#tensor_name = "encoder.layer.7.attention.self.query.weight"
dense_tensor = model.get_parameter(tensor_name)

#print(type(dense_tensor))
matrix = dense_tensor.detach().numpy()
print(matrix.shape)

results = []
dense_norm  = float(torch.norm(dense_tensor, p=1))

nrows = matrix.shape[0]
ncols = matrix.shape[1]

nm = [(2,4), (2,5), (2,7), (2,8), (2,10), (1,5), (2,20), (1,10), (1,16), (2,40)]

desired_sparsities = [1-n/m for n,m in nm]
tiles = [8,16,32,64,128]

n_init = 150


try:
    cache = functools.cache
except AttributeError:
    cache = functools.lru_cache(maxsize=None)


@cache
def group_n_m2(dense_shape, dense_dtype, n, m, tileM):
    nrows = dense_shape[0]
    ncols = dense_shape[1]

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

        extern "C" void func({dtype}* dense, {dtype}* sparse, float* masks){{
            int bm_m   = {nrows}/{tileM};
            int mcol_k = {ncols}/{m};
            int mcol_k_p = int_ceil({ncols},{m});
            int m_fixed = 4;

            std::vector<int> columns(bm_m*mcol_k_p*m_fixed, 0);

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
                        columns[bm_i*mcol_k_p*m_fixed + mcol_i*m_fixed + i] =
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
                            columns[bm_i*mcol_k_p*m_fixed + mcol_k*m_fixed + j] = j;
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
                                columns[bm_i*mcol_k_p*m_fixed + mcol_i*m_fixed + i] =
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
                                columns[bm_i*mcol_k_p*m_fixed + mcol_i*m_fixed + i] =
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
    ]
    return lib.func

impl_builder = (
        group_n_m2
    )


for tileM in tiles:
    for n,m in nm:
        inp = torch.from_numpy(matrix).abs()
        sparse = inp.clone()
        masks = torch.zeros(matrix.shape)

        func = impl_builder(
                inp.shape,
                inp.dtype,
                n,
                m,
                tileM
        )

        func(inp.data_ptr(), sparse.data_ptr(), masks.data_ptr())

        group_n_m_tensor = sparse.numpy()
        nm_norm =  float(torch.norm(torch.from_numpy(group_n_m_tensor), p=1))

        results.append([str(n)+':'+str(m), str(tileM)+':N:M', 1-n/m, nm_norm/dense_norm])


def nmSparsifier(tensor, n, m):
    num_groups = tensor.numel() // m
    # N:M sparsity for linear layers
    tensor_temp = tensor.detach().abs().reshape(num_groups, m)
    index = torch.argsort(tensor_temp, dim=1)[:, : int(m - n)]
    w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)

    return w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)

def vector_groups(matrix, cols, m, k):
    # Sizes
    input_shape = matrix.shape
    rows, cols = input_shape[0], input_shape[1]
    d_rows, d_cols = k, 1
    subm_rows, subm_cols = rows-d_rows+1, cols-d_cols+1

    # Index grids
    ii, jj = np.meshgrid(range(0, subm_rows, d_rows), range(0, subm_cols, d_cols), indexing='ij')
    d_ii, d_jj = np.meshgrid(range(d_rows), range(d_cols), indexing='ij')

    # Add indices
    subm_ii = ii[:, :, np.newaxis, np.newaxis] + d_ii
    subm_jj = jj[:, :, np.newaxis, np.newaxis] + d_jj

    # Make submatrices tensor
    subm = matrix[subm_ii, subm_jj]

    subm_sum = np.sum(subm, axis=(2, 3))

    top_idx = np.argpartition(subm_sum.flatten(), -m)[-m:]  # Indices not sorted
    size = subm_sum.shape
    # Get row and column
    top_row = top_idx // size[1]
    top_col = top_idx % size[1]

    result = np.stack([top_row*d_rows, top_col*d_cols], axis=-1)

    mask = np.zeros((matrix.shape[0], matrix.shape[1]))
    for r in result:
        mask[r[0]:r[0]+d_rows, r[1]:r[1]+d_cols] = 1

    return mask


tensor_size = matrix.shape[0]*matrix.shape[1]

for v in [2,4,8,16,32]:
    #for desired_sparsity in desired_sparsities:
    for n,m in nm:
        desired_sparsity = 1-n/m
        elems = int(tensor_size*(1-desired_sparsity))

        masks = vector_groups(np.absolute(matrix), matrix.shape[1], elems//v, v)
        vector_wise_tensor = matrix*masks
        norm = float(torch.norm(torch.from_numpy(vector_wise_tensor), p=1))
        results.append([str(n)+':'+str(m), 'vw_'+str(v), desired_sparsity, norm/dense_norm])


def create_mask_threshold_criterion(tensor, threshold):
    with torch.no_grad():
        mask = torch.gt(torch.abs(tensor), threshold).type(tensor.type())
        return mask

def level(matrix, desired_sparsity):
    tensor = torch.from_numpy(matrix)
    bottomk, _ = torch.topk(tensor.abs().view(-1),
                                int(desired_sparsity * tensor.numel()),
                                largest=False,
                                sorted=True)
    threshold = bottomk.data[-1]
    mask = create_mask_threshold_criterion(tensor, threshold)
    return mask


#for desired_sparsity in desired_sparsities:
for n,m in nm:
    desired_sparsity = 1-n/m
    masks = level(np.absolute(matrix), desired_sparsity)

    magnitude_tensor = matrix*masks.detach().numpy()
    norm = float(torch.norm(torch.from_numpy(magnitude_tensor), p=1))

    results.append([str(n)+':'+str(m), "ideal", desired_sparsity, norm/dense_norm])


#from grouped_nm_tensor import GroupedNMTensor, PerfectNMTensor
for n,m in nm:
    #perfect_nm_tensor = PerfectNMTensor.from_dense(dense_tensor, n=n, m=m, sparse_dim=0)
    inp = torch.from_numpy(matrix).abs().clone()
    sparse = inp.clone()
    mask = torch.zeros(matrix.shape)
    func = impl_builder(
            inp.shape,
            inp.dtype,
            n,
            m,
            1
    )
    func(inp.data_ptr(), sparse.data_ptr(), mask.data_ptr())
    #sparse = nmSparsifier(dense_tensor, n, m)

    group_n_m_tensor = sparse.numpy()
    #pnm_norm = float(torch.norm(group_n_m_tensor, p=1))
    pnm_norm =  float(torch.norm(torch.from_numpy(group_n_m_tensor), p=1))

    results.append([str(n)+":"+str(m), '1:N:M', 1-n/m, pnm_norm/dense_norm])


print("******** results *********")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(results, columns=['density', 'name', 'sparsity', 'energy'])

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme()

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#df = pd.DataFrame(results, columns=['density', 'name', 'sparsity', 'energy'])

df = df[(df.density=='2:4') | (df.density=='2:5') | (df.density=='2:8') | (df.density=='2:10')| (df.density=='2:20') | (df.density=='2:40')]

sns.light_palette("seagreen", as_cmap=True)
pal = sns.color_palette("crest", n_colors=6)
pal2 = sns.color_palette("flare", 5)
pal3 = sns.color_palette("ch:start=.2,rot=-.3", 5)

pruners = ['ideal', '1:N:M', '16:N:M', '32:N:M', '64:N:M', '128:N:M', 'vw_4', 'vw_8', 'vw_16', 'vw_32']

colors = {'ideal':'k',
          '1:N:M':pal.as_hex()[0],
          '16:N:M':pal.as_hex()[2],
          '32:N:M':pal.as_hex()[3],
          '64:N:M':pal.as_hex()[4],
          '128:N:M':pal.as_hex()[5],
          'vw_4':pal2.as_hex()[3],
          'vw_8':pal2.as_hex()[2],
          'vw_16':pal2.as_hex()[1],
          'vw_32':pal2.as_hex()[0]}

df['sparsity'] = df['sparsity'].astype({'sparsity':'string'})
df['density'] = df['density'].astype({'density':'string'})
df['name'] = df['name'].astype('category')
df['name'] = df.name.cat.set_categories(pruners)
df['name'] = df['name'].astype('category')

mapping = {pruner: i for i, pruner in enumerate(pruners)}

#g = sns.catplot(data=df, kind='bar', x='density', y='energy', hue='name', aspect=1.7)
g = sns.catplot(data=df, kind='bar', x='density', y='energy', hue='name', palette=colors, aspect=1.7)
g._legend.remove()
plt.xlabel('sparsity [%] (N:M)')
plt.ylabel('Energy')
plt.legend()
plt.title('BERT-base encoder.layer.8.attention.self.query.weight: 768x768')
plt.savefig('result/energy.png', bbox_inches = 'tight')

