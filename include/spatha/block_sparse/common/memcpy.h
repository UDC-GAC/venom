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

#include "base.h"

namespace spatha {

namespace my_pipeline {


// device function to convert shared memory address into unsigned format
DEVICE_INLINE unsigned get_smem_ptr(const void *ptr)
{
// #if (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
// #else
//     return __nvvm_get_smem_pointer(ptr);
// #endif
}

DEVICE_INLINE
void copy_and_sync(unsigned *dst, const unsigned *src, int size)
{
    for (int i = threadIdx.x*4; i < size; i+=blockDim.x*4)
    {
        //dst[i] = src[i];
        (*(float4*)(dst+i)) = (*(float4*)(src+i));
    }
    __syncthreads();
}

template<int SizeInBytes> DEVICE_INLINE
void cp_async_pred_zfill(void * smem_ptr, void const * gmem_ptr,
    const bool pred_guard = true, const bool zfill = false)
{
  unsigned smem_int_ptr = get_smem_ptr(smem_ptr);
  int src_in_bytes = (zfill ? 0 : SizeInBytes );

  asm volatile (
    "{\n"
    "  .reg .pred p;\n"
    "  setp.ne.b32 p, %0, 0;\n"
    "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
    "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
  );
  /* asm volatile (
    "{\n"
    "  .reg .pred p;\n"
    "  setp.ne.b32 p, %0, 0;\n"
    "  @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
    "}\n" ::"r"((int)pred_guard),
    "r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
  ); */
}

template<int tile_size, int nthreads> DEVICE_INLINE
void cp_async_tile(uint* smem_ptr, const uint* gmem_ptr, const int valid_size) {
        // assume zfill_size % 4 == 0;
    for (int i = threadIdx.x; i < ROUND_UP(tile_size, nthreads); i += nthreads) {
        bool valid = i < tile_size;
        int src_in_bytes = (i < valid_size) ? 4 : 0;
        unsigned dst = get_smem_ptr(smem_ptr + i);
        const uint *src = gmem_ptr + i;
        asm volatile (
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
            "}\n" ::"r"((int)valid),
            "r"(dst), "l"(src), "n"(4), "r"(src_in_bytes)
        );
    }
}


template<int NStage, bool UseMinSync> struct Pipeline;


template<int NStage>
struct Pipeline<NStage, false> {
    //static_assert(NStage>1);

    DEVICE_INLINE
    void acquire_writer(){
    }
    DEVICE_INLINE
    void commit_stage() {
        asm volatile("cp.async.commit_group;\n"::);
    }
    DEVICE_INLINE
    void acquire_reader() {
        asm volatile ("cp.async.wait_group %0;\n"::"n"(NStage-1));
        __syncthreads();
    }
    DEVICE_INLINE
    void release_reader() {
        __syncthreads();
    }
};

template<int NStage>
struct Pipeline<NStage, true> {
    //static_assert(NStage>1);
    int ahead_stage = 0;

    DEVICE_INLINE
    void acquire_writer(){
        if (ahead_stage == NStage -1) {
            asm volatile ("cp.async.wait_group %0;\n"::"n"(NStage-2));
            __syncthreads();
        }
    }
    DEVICE_INLINE
    void commit_stage() {
        asm volatile("cp.async.commit_group;\n"::);
        ahead_stage++;
    }
    DEVICE_INLINE
    void acquire_reader() {
    }
    DEVICE_INLINE
    void release_reader() {
        ahead_stage--;
    }
};

} // namespace my_pipeline
}