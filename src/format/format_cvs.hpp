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

#ifndef FORMAT_CVS_H
#define FORMAT_CVS_H

#include "format_cxx.hpp"
template<class T>
class Format_cvs: public Format_cxx<T> {
    public:
        Format_cvs(){};
        ~Format_cvs();
        void init(int nrow_, int ncol_, int nnz_, float density_, unsigned seed_, bool row_permute_, int brow_, int bcol_, int mbrow=0, int bm=1);
        int get_vec_length();
        void to_String();
        std::vector<T>& to_dense() override;
        void change_v_length(int wm);
    //private:
};

#endif