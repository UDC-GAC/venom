#pragma once
#include <vector>

namespace spatha {

void random_mask(int nrow, int ncol, int brow, int bcol, float block_density,
    int  &nnzb, int &padded_nnzb,
    std::vector<int> &indptr, std::vector<int> &indices,
    int block_divisor, int block_padding, unsigned seed) ;

void random_mask(int nrow, int ncol, int brow, int bcol, float block_density,
    int &nnzb, std::vector<int> &indptr, std::vector<int> &indices, unsigned seed);

}
