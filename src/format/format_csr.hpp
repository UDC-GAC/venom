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

#ifndef FORMAT_CSR_H
#define FORMAT_CSR_H

#include "format_cxx.hpp"
template<class T>
class Format_csr: public Format_cxx<T> {
    public:
        Format_csr(){};
        ~Format_csr();
        void init(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_, bool row_permute_, int brow_=1, int bcol_=1, int mbrow=0, int bm_=0); //
        //void init_from_file();
        void to_String();
        std::vector<T>& to_dense();
};

#endif