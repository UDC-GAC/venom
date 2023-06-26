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

#ifndef SPATHA_SPMM_H
#define SPATHA_SPMM_H

#include "spmm.hpp"

//#pragma once
#include "../../include/spatha/block_sparse/spmm/spmm_op.h"
#include "../../include/spatha/block_sparse/spmm/spmm_library_decl.h"

#include "../format/format_sr_nm.hpp"

using namespace std;
using namespace spatha;

template<class T, class T2>
class Spatha: public Spmm_CNN<T,T2> {
    public:
        // Empty virtual destructor for proper cleanup
        ~Spatha();
        Spatha(Format_sr_nm<T>& f, Dataset<T,T2> &d, cudaDataType_t S);
        Spatha(Dataset<T,T2> &d, cudaDataType_t S, cusparseComputeType C_spmm);
        std::vector<T>& get_hB();
        std::vector<T2>& get_hC();
        std::vector<T2>& get_result();
        std::vector<T>& get_sparse();
        //float spmm(int times);
        float spmm(int times, int bm_=0, int bn_=0, int bk_=0, int wm_=0, int wn_=0, int wk_=0, int mm_=0, int mn_=0, int mk_=0, int nstage_=0);

    private:
        spatha::BlockwiseSpMatrix<T> spmat;
        std::vector<T2> result;

        std::map<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>, spatha::SpmmNMInitFn_t> init_fn = {
        {std::make_tuple(32, 16, 16, 32, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 16, 16, 32, 16, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 64, 16, 32, 64, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 16, 32, 64, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 16, 32, 64, 16, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 16, 32, 64, 16, 16, 8, 16, 3)}, //
        {std::make_tuple(32, 64, 16, 32, 64, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 16, 32, 64, 16, 16, 8, 16, 4)}, //

        {std::make_tuple(32, 64, 128, 32, 64, 128, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 128, 32, 64, 128, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 128, 16, 32, 64, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 16, 32, 64, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 128, 16, 32, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 16, 32, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 128, 16, 32, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 16, 32, 16, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 128, 16, 32, 128, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 16, 32, 128, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 64, 16, 32, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 16, 32, 16, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 16, 32, 16, 16, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 16, 32, 16, 16, 16, 8, 16, 3)}, //
        {std::make_tuple(32, 64, 16, 32, 16, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 16, 32, 16, 16, 16, 8, 16, 4)}, //

        {std::make_tuple(64, 64, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 32, 64, 32, 16, 8, 32, 2)},
        {std::make_tuple(64, 64, 32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 32, 64, 32, 16, 8, 32, 3)},
        {std::make_tuple(64, 64, 32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 32, 64, 32, 16, 8, 32, 4)},
        {std::make_tuple(64, 64, 32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 64, 32, 16, 8, 32, 2)},
        {std::make_tuple(64, 64, 32, 64, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 64, 32, 16, 8, 32, 3)},
        {std::make_tuple(64, 64, 32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 64, 32, 16, 8, 32, 4)},

        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 64, 32, 16, 8, 32, 2)},
        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 64, 32, 16, 8, 32, 3)},
        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 64, 32, 16, 8, 32, 4)},


        {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 32, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 32, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 32, 32, 32, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 32, 32, 32, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 32, 32, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 64, 32, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 64, 32, 32, 64, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 64, 32, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 64, 32, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 64, 32, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 64, 64, 32, 16, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 16, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 64, 32, 16, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 16, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 64, 32, 16, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 16, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 64, 64, 32, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 64, 32, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 64, 32, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 64, 32, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 64, 128, 32, 64, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 128, 32, 64, 128, 16, 8, 32, 2)}, //

        {std::make_tuple(32, 128, 128, 32, 64, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 128, 32, 64, 128, 16, 8, 32, 2)}, //

        {std::make_tuple(32, 64, 32, 32, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 32, 32, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 32, 32, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 64, 32, 32, 16, 32, 16, 8, 32, 4)}, //

        /* {std::make_tuple(32, 256, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 256, 32, 32, 64, 32, 16, 8, 32, 2)}, // */

        {std::make_tuple(32, 128, 64, 32, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 64, 32, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 128, 64, 32, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 64, 32, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 128, 64, 32, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 64, 32, 64, 64, 16, 8, 32, 4)}, //
        {std::make_tuple(32, 128, 128, 32, 128, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 128, 32, 128, 128, 16, 8, 32, 2)}, //
        /* {std::make_tuple(32, 128, 128, 32, 128, 128, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 128, 32, 128, 128, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 128, 128, 32, 128, 128, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 32, 128, 128, 32, 128, 128, 16, 8, 32, 4)}, // */


        {std::make_tuple(64, 32, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 32, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64, 32, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 64, 32, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 32, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 64, 32, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64, 64, 16, 64, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 16, 64, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(64, 64, 16, 64, 32, 16, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 16, 64, 32, 16, 16, 8, 16, 3)}, //
        {std::make_tuple(64, 64, 16, 64, 32, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 16, 64, 32, 16, 16, 8, 16, 4)}, //

        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 32, 32, 16, 8, 16, 3)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 32, 64, 32, 32, 16, 8, 16, 4)}, //

        {std::make_tuple(64, 64, 64, 32, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 64, 64, 32, 64, 64, 16, 8, 32, 2)}, //

        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 32, 32, 16, 8, 32, 4)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 5), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 32, 32, 16, 8, 32, 5)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 6), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 32, 32, 16, 8, 32, 6)}, //

        {std::make_tuple(128, 64, 64, 64, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 64, 64, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 64, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 64, 64, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 64, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 64, 64, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 128, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 128, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 128, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 128, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 128, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 128, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 16, 128, 32, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 16, 128, 32, 16, 16, 8, 16, 4)}, //

        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 128, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 64, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 64, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 64, 32, 32, 16, 8, 32, 4)}, //


        /* {std::make_tuple(128, 32, 128, 128, 32, 128, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 32, 128, 128, 32, 128, 16, 8, 32, 4)}, // */

        {std::make_tuple(128, 64, 16, 128, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 16, 128, 32, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(64, 32, 16, 32, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 32, 16, 32, 32, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(256, 64, 16, 64, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 256, 64, 16, 64, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(256, 32, 16, 64, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 256, 32, 16, 64, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 256, 32, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 256, 32, 32, 64, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(256, 64, 16, 64, 64, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 256, 64, 16, 64, 64, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(64, 16, 16, 64, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 16, 16, 64, 16, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(128, 16, 16, 64, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 16, 16, 64, 16, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(128, 16, 16, 128, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 16, 16, 128, 16, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(128, 32, 32, 128, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 32, 32, 128, 32, 32, 16, 8, 32, 2)}, //

        {std::make_tuple(128, 128, 32, 128, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 128, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 128, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 128, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 128, 32, 128, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64, 32, 128, 64, 32, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 64, 32, 128, 64, 32, 128, 16, 8, 32, 2)}, //
        /* {std::make_tuple(64, 32, 128, 64, 32, 128, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 64, 32, 128, 64, 32, 128, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 32, 128, 64, 32, 128, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 64, 32, 128, 64, 32, 128, 16, 8, 32, 4)}, // */

        {std::make_tuple(128, 32, 32, 32, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 32, 32, 32, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 32, 32, 32, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 32, 32, 32, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 32, 32, 32, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 32, 32, 32, 16, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 256, 32, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 256, 32, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 64, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 256, 64, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 64, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 256, 64, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 64, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 256, 64, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 128, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,256, 128, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 128, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,256, 128, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 128, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,256, 128, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 128, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,256, 128, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 128, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,256, 128, 32, 32, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 128, 32, 32, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,256, 128, 32, 32, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 64, 64, 64, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,256, 64, 64, 64, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 64, 64, 64, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,256, 64, 64, 64, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 64, 64, 64, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,256, 64, 64, 64, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 64, 64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,256, 64, 64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 64, 64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,256, 64, 64, 64, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 64, 64, 64, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,256, 64, 64, 64, 64, 64, 16, 8, 32, 4)}, //

	/////////////
        {std::make_tuple(64, 64, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,64, 64, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64, 64, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,64, 64, 32, 32, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 64, 32, 32, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,64, 64, 32, 32, 32, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128, 64, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 32, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 32, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 32, 64, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128, 32, 64, 64, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 32, 64, 64, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 32, 64, 64, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 32, 64, 64, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 32, 64, 64, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128, 32, 64, 64, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 64, 64, 16, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 64, 64, 16, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 64, 16, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 64, 64, 16, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 64, 16, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 64, 64, 16, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 64, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 64, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 64, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 64, 16, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 32, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 32, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 32, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 32, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 32, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128, 64, 32, 32, 16, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128,32, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128,32, 32, 32, 32, 32, 16, 8, 32, 2)}, //
	{std::make_tuple(128,32, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128,32, 32, 32, 32, 32, 16, 8, 32, 3)}, //

	//{std::make_tuple(128,64, 32, 16, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128,64, 32, 16, 64, 32, 16, 8, 32, 2)}, //
	//{std::make_tuple(128,64, 32, 16, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128,64, 32, 16, 64, 32, 16, 8, 32, 3)}, //
	//{std::make_tuple(128,64, 32, 16, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128,64, 32, 16, 64, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128,64, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128,64, 32, 32, 32, 32, 16, 8, 32, 2)}, //
	{std::make_tuple(128,64, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init, 128,64, 32, 32, 32, 32, 16, 8, 32, 3)}, //
	{std::make_tuple(128,64, 32, 32, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128,64, 32, 32, 32, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128,128,32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128,128,32, 32, 64, 32, 16, 8, 32, 2)}, //
	{std::make_tuple(128,128,32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128,128,32, 32, 64, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128,128,32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init, 128,128,32, 64, 32, 32, 16, 8, 32, 2)}, //
	{std::make_tuple(128,128,32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init, 128,128,32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 64, 32, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 128, 64, 32, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 64, 32, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 128, 64, 32, 32, 64, 16, 8, 32, 3)}, //

        {std::make_tuple(128, 128, 64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 128, 64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 128, 64, 64, 64, 64, 16, 8, 32, 3)}, //

        {std::make_tuple(128, 64, 32, 128, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 128, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 128, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 128, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 128, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 128, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 64, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 64, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 64, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 32, 64, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 32, 64, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128,128, 32, 64, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 64, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128,128, 32, 64, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 128, 32, 64, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128,128, 32, 64, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 64, 64, 128, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128,128, 64, 64, 128, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 64, 64, 128, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128,128, 64, 64, 128, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128,128, 64, 64, 128, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128,128, 64, 64, 128, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 64, 64, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 64, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 64, 64, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 64, 128, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 64, 128, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 128, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 64, 128, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 128, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,128, 64, 64, 128, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 64, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 64, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 64, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 64, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,64,128,64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,64,128,64, 64, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,64, 64, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,64,128,64, 64, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 32, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 32, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 32, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,32, 32, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 32, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 32, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 32, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 32, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 32, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,32, 64, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 64, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 64, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 64, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 64, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Init,64,128,32, 64, 128, 32, 16, 8, 32, 4)}, //

        };

        std::map<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>, spatha::SpmmNMExecFn_t> exec_fn = {
        /* {std::make_tuple(32, 32, 16, 32, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 32, 16, 32, 32, 16, 16, 8, 16, 2)}, // */
        {std::make_tuple(32, 16, 16, 32, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 16, 16, 32, 16, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 64, 16, 32, 64, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 16, 32, 64, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 16, 32, 64, 16, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 16, 32, 64, 16, 16, 8, 16, 3)}, //
        {std::make_tuple(32, 64, 16, 32, 64, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 16, 32, 64, 16, 16, 8, 16, 4)}, //

        {std::make_tuple(32, 64, 128, 32, 64, 128, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 128, 32, 64, 128, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 128, 16, 32, 128, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 16, 32, 128, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 128, 16, 32, 64, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 16, 32, 64, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 128, 16, 32, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 16, 32, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 128, 16, 32, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 16, 32, 16, 16, 16, 8, 16, 2)}, //


        {std::make_tuple(32, 64, 16, 32, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 16, 32, 16, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 16, 32, 16, 16, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 16, 32, 16, 16, 16, 8, 16, 3)}, //
        {std::make_tuple(32, 64, 16, 32, 16, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 16, 32, 16, 16, 16, 8, 16, 4)}, //

        /* {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 1), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 32, 64, 16, 8, 32, 1)}, //
        {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 16, 1), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 32, 64, 16, 8, 16, 1)}, //
        {std::make_tuple(32, 64, 32, 32, 32, 32, 16, 8, 32, 1), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 32, 32, 16, 8, 32, 1)}, //
        {std::make_tuple(32, 32, 32, 32, 32, 32, 16, 8, 32, 1), NAME_FUNC(spatha::SpmmNM, Exec, 32, 32, 32, 32, 32, 32, 16, 8, 32, 1)}, //
        {std::make_tuple(32, 32, 32, 32, 32, 32, 16, 8, 16, 1), NAME_FUNC(spatha::SpmmNM, Exec, 32, 32, 32, 32, 32, 32, 16, 8, 16, 1)}, //
        {std::make_tuple(32, 64, 64, 32, 64, 64, 16, 8, 32, 1), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 64, 64, 16, 8, 32, 1)}, // */

        {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 32, 64, 16, 8, 32, 2)}, //
        /* {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 32, 64, 16, 8, 32, 3)}, // */

        {std::make_tuple(32, 32, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 32, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 32, 32, 32, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 32, 32, 32, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 32, 32, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 64, 32, 16, 8, 16, 2)}, //

        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 32, 64, 16, 8, 16, 2)}, //

        {std::make_tuple(64, 64, 64, 32, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 64, 32, 64, 64, 16, 8, 32, 2)}, //

        /* {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 32, 1), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 32, 64, 16, 8, 32, 1)}, // */
        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 128, 64, 32, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 64, 64, 32, 16, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 16, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 64, 32, 16, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 16, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 64, 32, 16, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 16, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 64, 64, 32, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 64, 32, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 64, 32, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 128, 128, 32, 64, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 128, 32, 64, 128, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 64, 32, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 64, 32, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(32, 64, 32, 32, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 64, 32, 32, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 64, 32, 32, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 16, 32, 16, 8, 32, 4)}, //

        /* {std::make_tuple(32, 256, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 256, 32, 32, 64, 32, 16, 8, 32, 2)}, // */

        {std::make_tuple(32, 64, 128, 32, 64, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 128, 32, 64, 128, 16, 8, 32, 2)}, //

        {std::make_tuple(32, 128, 64, 32, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(32, 128, 64, 32, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 128, 64, 32, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 64, 32, 64, 64, 16, 8, 32, 4)}, //
        {std::make_tuple(32, 128, 128, 32, 128, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 128, 32, 128, 128, 16, 8, 32, 2)}, //
        /* {std::make_tuple(32, 128, 128, 32, 128, 128, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 128, 32, 128, 128, 16, 8, 32, 3)}, //
        {std::make_tuple(32, 128, 128, 32, 128, 128, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 128, 128, 32, 128, 128, 16, 8, 32, 4)}, // */


        {std::make_tuple(64, 32, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 32, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64, 32, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 64, 32, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 32, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 64, 32, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64, 64, 16, 64, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 16, 64, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(64, 64, 16, 64, 32, 16, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 16, 64, 32, 16, 16, 8, 16, 3)}, //
        {std::make_tuple(64, 64, 16, 64, 32, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 16, 64, 32, 16, 16, 8, 16, 4)}, //

        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 16, 3), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 32, 32, 16, 8, 16, 3)}, //
        {std::make_tuple(64, 64, 32, 64, 32, 32, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 32, 32, 16, 8, 16, 4)}, //

        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 32, 32, 16, 8, 32, 4)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 5), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 32, 32, 16, 8, 32, 5)}, //
        {std::make_tuple(128, 64, 32, 64, 32, 32, 16, 8, 32, 6), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 32, 32, 16, 8, 32, 6)}, //

        {std::make_tuple(128, 64, 64, 64, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 64, 64, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 64, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 64, 64, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 64, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 64, 64, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 128, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 128, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 128, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 128, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 128, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 128, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 16, 128, 32, 16, 16, 8, 16, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 16, 128, 32, 16, 16, 8, 16, 4)}, //

        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 128, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 64, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 64, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        /* {std::make_tuple(128, 32, 128, 128, 32, 128, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 32, 128, 128, 32, 128, 16, 8, 32, 4)}, // */

        {std::make_tuple(128, 64, 16, 128, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 16, 128, 32, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(64, 32, 16, 32, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 32, 16, 32, 32, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(256, 64, 16, 64, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 256, 64, 16, 64, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(256, 32, 16, 64, 32, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 256, 32, 16, 64, 32, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 256, 32, 32, 64, 32, 32, 16, 8, 16, 2)}, //
        {std::make_tuple(256, 64, 16, 64, 64, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 256, 64, 16, 64, 64, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(64, 16, 16, 64, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 16, 16, 64, 16, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(128, 16, 16, 64, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 16, 16, 64, 16, 16, 16, 8, 16, 2)}, //
        {std::make_tuple(128, 16, 16, 128, 16, 16, 16, 8, 16, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 16, 16, 128, 16, 16, 16, 8, 16, 2)}, //

        {std::make_tuple(128, 32, 32, 128, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 32, 32, 128, 32, 32, 16, 8, 32, 2)}, //

        {std::make_tuple(128, 128, 32, 128, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 128, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 128, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 128, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 128, 32, 128, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 128, 32, 128, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64, 32, 128, 64, 32, 128, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 32, 128, 64, 32, 128, 16, 8, 32, 2)}, //
        /* {std::make_tuple(64, 32, 128, 64, 32, 128, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 64, 32, 128, 64, 32, 128, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 32, 128, 64, 32, 128, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 64, 32, 128, 64, 32, 128, 16, 8, 32, 4)}, // */

        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 256, 32, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 256, 32, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 32, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 256, 32, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 64, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 256, 64, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 64, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 256, 64, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 64, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 256, 64, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 128, 32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,256, 128, 32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 128, 32, 64, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,256, 128, 32, 64, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 128, 32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,256, 128, 32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 128, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,256, 128, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 128, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,256, 128, 32, 32, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 128, 32, 32, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,256, 128, 32, 32, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 64, 64, 64, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,256, 64, 64, 64, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 64, 64, 64, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,256, 64, 64, 64, 32, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 64, 64, 64, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,256, 64, 64, 64, 32, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(256, 64, 64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,256, 64, 64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(256, 64, 64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,256, 64, 64, 64, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(256, 64, 64, 64, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,256, 64, 64, 64, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(64, 64, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 32, 64, 32, 16, 8, 32, 2)},
        {std::make_tuple(64, 64, 32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 32, 64, 32, 16, 8, 32, 3)},
        {std::make_tuple(64, 64, 32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 32, 64, 32, 16, 8, 32, 4)},
        {std::make_tuple(64, 64, 32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 64, 32, 16, 8, 32, 2)},
        {std::make_tuple(64, 64, 32, 64, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 64, 32, 16, 8, 32, 3)},
        {std::make_tuple(64, 64, 32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 64, 64, 32, 64, 64, 32, 16, 8, 32, 4)},


        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 64, 32, 16, 8, 32, 2)},
        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 64, 32, 16, 8, 32, 3)},
        {std::make_tuple(32, 64, 32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 32, 64, 32, 32, 64, 32, 16, 8, 32, 4)},

	/////////////
        {std::make_tuple(64, 64, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,64, 64, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64, 64, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,64, 64, 32, 32, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64, 64, 32, 32, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,64, 64, 32, 32, 32, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128, 64, 32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 32, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 32, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 32, 64, 32, 16, 8, 32, 4)}, //

	{std::make_tuple(128, 32, 64, 64, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 32, 64, 64, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 32, 64, 64, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 32, 64, 64, 32, 64, 16, 8, 32, 3)}, //
	{std::make_tuple(128, 32, 64, 64, 32, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128, 32, 64, 64, 32, 64, 16, 8, 32, 4)}, //

 	{std::make_tuple(128, 64, 64, 64, 16, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 64, 64, 16, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 64, 16, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 64, 64, 16, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 64, 16, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 64, 64, 16, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 64, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 64, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 64, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 64, 16, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128,32, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128,32, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128,32, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128,32, 32, 32, 32, 32, 16, 8, 32, 3)}, //

        //{std::make_tuple(128,64, 32, 16, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128,64, 32, 16, 64, 32, 16, 8, 32, 2)}, //
        //{std::make_tuple(128,64, 32, 16, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128,64, 32, 16, 64, 32, 16, 8, 32, 3)}, //
        //{std::make_tuple(128,64, 32, 16, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128,64, 32, 16, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128,64, 32, 32, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128,64, 32, 32, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128,64, 32, 32, 32, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128,64, 32, 32, 32, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128,64, 32, 32, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128,64, 32, 32, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128,128,32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128,128,32, 32, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128,128,32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128,128,32, 32, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128,128,32, 64, 32, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128,128,32, 64, 32, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128,128,32, 64, 32, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128,128,32, 64, 32, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 64, 32, 32, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 128, 64, 32, 32, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 64, 32, 32, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 128, 64, 32, 32, 64, 16, 8, 32, 3)}, //

        {std::make_tuple(128, 128, 64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 128, 64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 128, 64, 64, 64, 64, 16, 8, 32, 3)}, //

        {std::make_tuple(128, 64, 32, 128, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 128, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 128, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 128, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 128, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 128, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 64, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 64, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 64, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 32, 64, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 32, 64, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128,128, 32, 64, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 32, 64, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128,128, 32, 64, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 128, 32, 64, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128,128, 32, 64, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 128, 64, 64, 128, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128,128, 64, 64, 128, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 128, 64, 64, 128, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128,128, 64, 64, 128, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 128, 64, 64, 128, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128,128, 64, 64, 128, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 64, 64, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 64, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 64, 64, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 64, 128, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 64, 128, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 64, 128, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 64, 128, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 64, 128, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,128, 64, 64, 128, 64, 64, 16, 8, 32, 4)}, //
//
        {std::make_tuple(64,128,32, 64, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 64, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 64, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 64, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 64, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 64, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,64, 64, 64, 64, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,64,128,64, 64, 64, 64, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,64, 64, 64, 64, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,64,128,64, 64, 64, 64, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,64, 64, 64, 64, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,64,128,64, 64, 64, 64, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,32, 32, 64, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 32, 64, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 32, 64, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 32, 64, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 32, 64, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 32, 64, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,32, 32, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 32, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 32, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 32, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 32, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 32, 128, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(64,128,32, 64, 128, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 64, 128, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(64,128,32, 64, 128, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 64, 128, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(64,128,32, 64, 128, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec,64,128,32, 64, 128, 32, 16, 8, 32, 4)}, //


        {std::make_tuple(128, 32, 32, 32, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 32, 32, 32, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 32, 32, 32, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 32, 32, 32, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 32, 32, 32, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 32, 32, 32, 16, 32, 16, 8, 32, 4)}, //

        {std::make_tuple(128, 64, 32, 32, 16, 32, 16, 8, 32, 2), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 32, 16, 32, 16, 8, 32, 2)}, //
        {std::make_tuple(128, 64, 32, 32, 16, 32, 16, 8, 32, 3), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 32, 16, 32, 16, 8, 32, 3)}, //
        {std::make_tuple(128, 64, 32, 32, 16, 32, 16, 8, 32, 4), NAME_FUNC(spatha::SpmmNM, Exec, 128, 64, 32, 32, 16, 32, 16, 8, 32, 4)}, //

	};

};

#endif
