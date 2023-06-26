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
#include "../common/library_util.h"

namespace spatha{

DECL_FUNC(SpmmNM, 32, 16, 16, 32, 16, 16, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 3);
DECL_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 4);

DECL_FUNC(SpmmNM, 32, 64, 128, 32, 64, 128, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 32, 128, 16, 32, 128, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 32, 128, 16, 32, 64, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 32, 128, 16, 32, 32, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 32, 128, 16, 32, 16, 16, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 3);
DECL_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 4);

/* DECL_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 1);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 16, 1);
DECL_FUNC(SpmmNM, 32, 64, 32, 32, 32, 32, 16, 8, 32, 1);
DECL_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 32, 1);
DECL_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 16, 1);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 1); */
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 32, 64, 32, 32, 32, 32, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 32, 64, 32, 32, 64, 32, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 16, 2);

//DECL_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 1);
DECL_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 32, 64, 128, 32, 64, 128, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 128, 128, 32, 64, 128, 16, 8, 32, 2);

DECL_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 4);

//DECL_FUNC(SpmmNM, 32, 256, 32, 32, 64, 32, 16, 8, 32, 2);

DECL_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 32, 128, 128, 32, 128, 128, 16, 8, 32, 2);
/* DECL_FUNC(SpmmNM, 32, 128, 128, 32, 128, 128, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 32, 128, 128, 32, 128, 128, 16, 8, 32, 4); */

DECL_FUNC(SpmmNM, 32, 64, 32, 32, 64, 32, 16, 8, 32, 2);


DECL_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 3);
DECL_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 4);

DECL_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 3);
DECL_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 4);

DECL_FUNC(SpmmNM, 64, 64, 64, 32, 64, 64, 16, 8, 32, 2);

DECL_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 5);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 6);

DECL_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 16, 128, 32, 16, 16, 8, 16, 4);

DECL_FUNC(SpmmNM, 128, 128, 32, 64, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 128, 32, 64, 64, 32, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 128, 128, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 128, 32, 64, 32, 32, 16, 8, 32, 4);

//DECL_FUNC(SpmmNM, 128, 32, 128, 128, 32, 128, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 16, 128, 32, 16, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 64, 32, 16, 32, 32, 16, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 256, 64, 16, 64, 32, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 256, 32, 16, 64, 32, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 256, 64, 16, 64, 64, 16, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 64, 16, 16, 64, 16, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 128, 16, 16, 64, 16, 16, 16, 8, 16, 2);
DECL_FUNC(SpmmNM, 128, 16, 16, 128, 16, 16, 16, 8, 16, 2);

DECL_FUNC(SpmmNM, 128, 32, 32, 128, 32, 32, 16, 8, 32, 2);

DECL_FUNC(SpmmNM, 128, 128, 32, 128, 64, 32, 16, 8, 32, 2);

DECL_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64, 32, 128, 64, 32, 128, 16, 8, 32, 2);
/* DECL_FUNC(SpmmNM, 64, 32, 128, 64, 32, 128, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64, 32, 128, 64, 32, 128, 16, 8, 32, 4); */

DECL_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 4);
DECL_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 4);

///
DECL_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 4);

//
DECL_FUNC(SpmmNM, 128,32, 32, 32, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128,32, 32, 32, 32, 32, 16, 8, 32, 3);

//DECL_FUNC(SpmmNM, 128,64, 32, 16, 64, 32, 16, 8, 32, 2);
//DECL_FUNC(SpmmNM, 128,64, 32, 16, 64, 32, 16, 8, 32, 3);
//DECL_FUNC(SpmmNM, 128,64, 32, 16, 64, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128,128,32, 32, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128,128,32, 32, 64, 32, 16, 8, 32, 4);

//DECL_FUNC(SpmmNM, 128,128,32, 64, 32, 32, 16, 8, 32, 2);
//DECL_FUNC(SpmmNM, 128,128,32, 64, 32, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 128, 64, 32, 32, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 128, 64, 32, 32, 64, 16, 8, 32, 3);

DECL_FUNC(SpmmNM, 128, 128, 64, 64, 64, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 128, 64, 64, 64, 64, 16, 8, 32, 3);

DECL_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 4);

//
DECL_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 4);

//
DECL_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 4);

DECL_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 2);
DECL_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 3);
DECL_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 4);

}
