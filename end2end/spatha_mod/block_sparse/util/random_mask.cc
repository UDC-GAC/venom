#include <set>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <random>
#include <chrono>
#include <cassert>
#include "random_mask.h"

namespace spatha {

void random_mask(int nrow, int ncol, int brow, int bcol, float block_density,
    int  &nnzb, int &padded_nnzb,
    std::vector<int> &indptr, std::vector<int> &indices,
    int block_divisor, int block_padding, unsigned seed)
{
    int X = nrow / brow;
    int Y = ncol / bcol;
    int nnzb_group = (int)(std::round(block_density * X * Y / block_divisor));
    nnzb = nnzb_group * block_divisor;

    std::vector<int> idx_group(X * (Y/block_divisor));
    std::iota(idx_group.begin(), idx_group.end(), 0);
    // unsigned seed = 2021;
    // if (timeseed) // use chrono to roll a seed
    //     seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(idx_group.begin(), idx_group.end(), std::default_random_engine(seed));
    std::vector<int> row_nnz_count(X, 0);
    for (auto it = idx_group.begin(); it != idx_group.begin() + nnzb_group; it++) {
        row_nnz_count[(*it) % X] += block_divisor;
    }

    std::vector<std::vector<int>> nz_list(X, std::vector<int>());
    for (int i = 0; i < X; i++) {
        int row_nnz = row_nnz_count[i];
        std::vector<int> cid(Y);
        std::iota(cid.begin(), cid.end(), 0);
        std::shuffle(cid.begin(), cid.end(), std::default_random_engine(seed +i+1));
        nz_list[i] = std::vector<int>(cid.begin(), cid.begin() + row_nnz);
    }
    int offset = 0, padded_offset = 0;
    indptr.clear();
    indices.clear();
    for (int i = 0; i < X; i++) {
        indptr.push_back(padded_offset);
        std::sort(nz_list[i].begin(), nz_list[i].end());
        std::copy(nz_list[i].begin(), nz_list[i].end(), std::back_inserter(indices));
        offset += nz_list[i].size();
        padded_offset += nz_list[i].size();
        while (padded_offset % block_padding != 0) {
            indices.push_back(*(nz_list[i].end() -1));
            padded_offset++;
        }
        // printf("%d\n", padded_offset);
    }
    assert (offset == nnzb);
    indptr.push_back(padded_offset);
    padded_nnzb = padded_offset;
}

void random_mask(int nrow, int ncol, int brow, int bcol, float block_density,
    int &nnzb, std::vector<int> &indptr, std::vector<int> &indices, unsigned seed)
{
    int nnzb_tmp, padded_nnzb_tmp;
    random_mask(nrow, ncol, brow, bcol, block_density, nnzb_tmp, padded_nnzb_tmp,
        indptr, indices, 1, 1, seed);
    assert(nnzb_tmp==padded_nnzb_tmp);
    nnzb = nnzb_tmp;
}

}