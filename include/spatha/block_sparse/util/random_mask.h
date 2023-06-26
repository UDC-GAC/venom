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
