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
#include "base.h"    // DEVICE_INLINE

namespace spatha {
namespace my_mma {

//#if __CUDA_ARCH__ >= 750

// device function to convert shared memory address into unsigned format
DEVICE_INLINE unsigned get_smem_ptr(const void *ptr)
{
// #if (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
// #else
//     return __nvvm_get_smem_pointer(ptr);
// #endif
}

#define __SM_32_INTRINSICS_DECL__ static __device__ __inline__

template<typename Shape> struct fragment_a_colmajor;
template<typename Shape> struct fragment_a_rowmajor;
template<typename Shape> struct fragment_b_rowmajor;
template<typename Shape> struct fragment_b_colmajor;
template<typename Shape, typename Accumulator> struct fragment_c;
template<typename Shape> struct fragment_a_sparse_cm;
template<typename Shape> struct fragment_b_sparse_rm;
template<typename Shape> struct fragment_meta_sparse;
template<typename Shape, typename Accumulator> struct fragment_c_sparse;

// *** 16x16x8 f32.f16 ***
template<> struct fragment_a_colmajor<ShapeBase<16, 16, 8 >> { uint x[2]; };
template<> struct fragment_b_rowmajor<ShapeBase<16, 16, 8 >> { uint x[2]; };
template<> struct fragment_c<ShapeBase<16, 16, 8 >, float> { float x[8] = {0};};
///#if  __CUDA_ARCH__ >= 800
// *** 16x16x16 f32.f16 ***
template<> struct fragment_a_colmajor<ShapeBase<16, 16, 16>> { uint x[4]; };
template<> struct fragment_a_rowmajor<ShapeBase<16, 16, 16>> { uint x[4]; };
template<> struct fragment_b_rowmajor<ShapeBase<16, 16, 16>> { uint x[4]; };
template<> struct fragment_b_colmajor<ShapeBase<16, 16, 16>> { uint x[4]; };
template<> struct fragment_c<ShapeBase<16, 16, 16>, float> { float x[8] = {0};};
template<> struct fragment_meta_sparse<ShapeBase<16, 16, 16>> { uint x[2]; };
// *** 16x16x16.sparse f32.f16 ***
//template<> struct fragment_a_sparse_cm<ShapeBase<16, 8, 16>> { uint x[2]; };
template<> struct fragment_a_sparse_cm<ShapeBase<16, 8, 16>> { uint x[4]; };
template<> struct fragment_b_sparse_rm<ShapeBase<16, 8, 16>> { uint x[2]; };
template<> struct fragment_c_sparse<ShapeBase<16, 8, 16>, float> { float x[4] = {0};};
template<> struct fragment_meta_sparse<ShapeBase<16, 8, 16>> { uint x[2]; };
// *** 16x16x32.sparse f32.f16 ***
template<> struct fragment_a_sparse_cm<ShapeBase<16, 8, 32>> { uint x[8]; };
template<> struct fragment_b_sparse_rm<ShapeBase<16, 8, 32>> { uint x[4]; };
//template<> struct fragment_c_sparse<ShapeBase<16, 8, 32>, float> { uint x[4] = {0};};
template<> struct fragment_c_sparse<ShapeBase<16, 8, 32>, float> { float x[8] = {0};};
template<> struct fragment_meta_sparse<ShapeBase<16, 8, 32>> { uint x[2]; };
///#endif

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<16, 16, 8 >> &a,
    const half *base, const int offset, const int ldm);
///#if  __CUDA_ARCH__ >= 800
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const half *base, const int offset, const int ldm);
void load_matrix_sync2(fragment_a_rowmajor<ShapeBase<16, 16, 16>> &a,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const half *base, const int offset, const int ldm);
DEVICE_INLINE
void load_meta_sync(fragment_meta_sparse<ShapeBase<16, 8, 16>> &a,
    const uint *base, const int offset);
DEVICE_INLINE
void load_meta_sync(uint * __restrict__ a,
    const uint *base, const int offset);
///#endif
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<16, 16, 8 >> &b,
    const half *base, const int offset, const int ldm);
///#if  __CUDA_ARCH__ >= 800
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const half *base, const int offset, const int ldm);
    template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_colmajor<ShapeBase<16, 16, 16>> &b,
    const half *base, const int offset, const int ldm);
void load_matrix_sync2(fragment_b_colmajor<ShapeBase<16, 16, 16>> &b,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_sparse_rm<ShapeBase<16, 8, 16>> &b,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_sparse_rm<ShapeBase<16, 8, 32>> &b,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void lds_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const half *base, const int offset);
template<class F> DEVICE_INLINE
void lds_matrix_sync(uint* __restrict__ a,
    const half *base, const int offset);

DEVICE_INLINE
void ldg_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const half *base);
DEVICE_INLINE
void ldg_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const half *base);

DEVICE_INLINE int div16(int numerator, int magic, int shift);
DEVICE_INLINE int mod16(int numerator, int div, int maxdiv);
DEVICE_INLINE int mad16(int a, int b, int c);

///#endif
DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 8 >, float> &d,
    const fragment_a_colmajor<ShapeBase<16, 16, 8 >> &a,
    const fragment_b_rowmajor<ShapeBase<16, 16, 8 >> &b,
    const fragment_c<ShapeBase<16, 16, 8 >, float> &c);
///#if  __CUDA_ARCH__ >= 800
DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 16>, float> &d,
    const fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const fragment_c<ShapeBase<16, 16, 16>, float> &c);
DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 16>, float> &d,
    const fragment_a_rowmajor<ShapeBase<16, 16, 16>> &a,
    const fragment_b_colmajor<ShapeBase<16, 16, 16>> &b,
    const fragment_c<ShapeBase<16, 16, 16>, float> &c);
DEVICE_INLINE void mma_sync_sparse(
    fragment_c_sparse<ShapeBase<16, 8, 16>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 16>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c);
DEVICE_INLINE void mma_sync_sparse(
    fragment_c_sparse<ShapeBase<16, 8, 32>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 32>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c);
DEVICE_INLINE void mma_sync_sparse_v(
    fragment_c_sparse<ShapeBase<16, 8, 16>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 16>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const fragment_meta_sparse<ShapeBase<16, 8, 32>> &e);
DEVICE_INLINE void mma_sync_sparse_v(
    fragment_c_sparse<ShapeBase<16, 8, 32>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 32>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const fragment_meta_sparse<ShapeBase<16, 8, 32>> &e);
///#endif
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<16, 16, 8 >, float> &c,
    const half *base, const int offset, const int ldm);
///#if  __CUDA_ARCH__ >= 800
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<16, 16, 16>, float> &c,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const half *base, const int offset, const int ldm, const int brow);
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void store_matrix_sync_128(const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void store_matrix_sync_128(const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const half *base, const int offset, const int ldm);

template<class F> DEVICE_INLINE
void store_matrix_sync2(const half2 (&out)[4], const half *base, const int offset, const int ldm, const int brow);
DEVICE_INLINE
void ld_shared_float4(const void *a, uint *r);
///#endif

// *** ldmatrix ***

template<bool trans, int num_reg, int nbit>
DEVICE_INLINE void ldmatrix(uint *dst, const void *src);

// *** implementation ***
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<16, 16, 8 >> &a,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane & 7;
    int col  = lane & 8;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 2, 16>(a.x, src);
}

///#if  __CUDA_ARCH__ >= 800
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = (lane & 7) + ((lane & 16)>>1);
    int col  = lane & 8;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 4, 16>(a.x, src);
}

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane & 7;
    int col  = lane & 8;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 2, 16>(a.x, src);
}

DEVICE_INLINE
void load_meta_sync(fragment_meta_sparse<ShapeBase<16, 8, 16>> &a,
    const uint *base, const int offset)
{
    //*(uint2*)&a.x[k*2] = *(uint2*)src;
    //*(uint2*)&a.x[0] = *(uint2*)base[offset];
    *( (uint2*)a.x ) = *( (uint2*) (base+offset) );
    //a.x[0] = base[offset];

    //__syncthreads();
}

DEVICE_INLINE
void load_meta_sync(uint * __restrict__ a,
    const uint *base, const int offset)
{
    *( (uint2*)a ) = *( (uint2*) (base+offset) );
}

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = (lane & 7) + ((lane & 16)>>1);
    //row*=2;
    //int row  = (lane & 7)*2 + (lane>>4);
    int col  = lane & 8;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 4, 16>(a.x, src);

    const half *src2 = base + f(offset + (row+16) * ldm + col);
    ldmatrix<true, 4, 16>(a.x+4, src2);
}

DEVICE_INLINE
void ldg_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const half *base){
    *( (float4*) a.x ) = *( (float4*) base );
}

DEVICE_INLINE
void ldg_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const half *base){
    *( (float4*) a.x ) = *( (float4*) base );
    //*( (float4*) (a.x+4) ) = *( (float4*) (base+4*8) ); //4(brow)*8(block_K/2/2)
    *( (float4*) (a.x+4) ) = *( (float4*) (base+4*8*8) ); //4(brow)*8(block_K/2/2)
}

template<class F> DEVICE_INLINE
void lds_matrix_sync(fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const half *base, const int offset){

    F f;
    const half *src = base + f(offset);

    *( (float4*) a.x ) = *( (float4*) src );
}

template<class F> DEVICE_INLINE
void lds_matrix_sync(uint* __restrict__ a,
    const half *base, const int offset){
    F f;

    *( (float4*) a ) = *( (float4*) (base + f(offset)) );
    *( (float4*) (a+4) ) = *( (float4*) (base + f(offset+128)) );
}
///#endif

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<16, 16, 8 >> &b,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane & 7;
    int col  = lane & 8;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 2, 16>(b.x, src);
}

//#if  __CUDA_ARCH__ >= 800
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = (lane & 15);
    int col  = ((lane & 16)>>1);
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 4, 16>(b.x, src);
}

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_colmajor<ShapeBase<16, 16, 16>> &b,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = (lane & 15);
    int col  = ((lane & 16)>>1);
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<false, 4, 16>(b.x, src);
}

template<class F> DEVICE_INLINE
void load_matrix_sync2(fragment_a_rowmajor<ShapeBase<16, 16, 16>> &a,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = (lane%8)*4 + ((lane/8)%2);
    int col  = (lane/16)*8;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<false, 4, 16>(a.x, src);
}

template<class F> DEVICE_INLINE
void load_matrix_sync2(fragment_b_colmajor<ShapeBase<16, 16, 16>> &b,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    //int row  = (lane%8)*2 + ((lane/8)%2);
    int row  = ((lane%8)/2)*4 + ((lane%16)/8)*2 + lane%2;
    int col  = (lane/16)*8;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<false, 4, 16>(b.x, src);
}


template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_sparse_rm<ShapeBase<16, 8, 16>> &b,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane & 15;
    int col  = 0;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 2, 16>(b.x, src);
}

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_sparse_rm<ShapeBase<16, 8, 32>> &b,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane;
    int col  = 0;
    F f;
    const half *src = base + f(offset + row * ldm + col);
    ldmatrix<true, 4, 16>(b.x, src);
}
//#endif

DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 8 >, float> &d,
    const fragment_a_colmajor<ShapeBase<16, 16, 8 >> &a,
    const fragment_b_rowmajor<ShapeBase<16, 16, 8 >> &b,
    const fragment_c<ShapeBase<16, 16, 8 >, float> &c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%8,%9}, {%10}, {%12,%13,%14,%15};\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%4, %5, %6, %7}, {%8,%9}, {%11}, {%16,%17,%18,%19};\n"
    :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),
     "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])
    : "r"(a.x[0]),  "r"(a.x[1]),
      "r"(b.x[0]),  "r"(b.x[1]),
      "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),
      "f"(c.x[4]),  "f"(c.x[5]),  "f"(c.x[6]),  "f"(c.x[7])
    );
}

//#if  __CUDA_ARCH__ >= 800
DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 16>, float> &d,
    const fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const fragment_c<ShapeBase<16, 16, 16>, float> &c)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%8, %9,%10,%11}, {%12,%13}, {%16,%17,%18,%19};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%4, %5, %6, %7}, {%8, %9,%10,%11}, {%14,%15}, {%20,%21,%22,%23};\n"
    :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),
     "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])
    : "r"(a.x[0]),  "r"(a.x[1]),  "r"(a.x[2]),  "r"(a.x[3]),
      "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
      "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),
      "f"(c.x[4]),  "f"(c.x[5]),  "f"(c.x[6]),  "f"(c.x[7])
    );
}

DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 16>, float> &d,
    const fragment_a_rowmajor<ShapeBase<16, 16, 16>> &a,
    const fragment_b_colmajor<ShapeBase<16, 16, 16>> &b,
    const fragment_c<ShapeBase<16, 16, 16>, float> &c)
{
    /* if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0){
        half* data = (half*) a.x;
        printf("%.2f %.2f %.2f %.2f \n", __half2float(data[0]), __half2float(data[2]), __half2float(data[1]), __half2float(data[3]));
        printf("%.2f %.2f %.2f %.2f \n", __half2float(b.x[0]), __half2float(b.x[2]), __half2float(b.x[1]), __half2float(b.x[3]));
    } */

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%8, %9,%10,%11}, {%12,%13}, {%16,%17,%18,%19};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%4, %5, %6, %7}, {%8, %9,%10,%11}, {%14,%15}, {%20,%21,%22,%23};\n"
    :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[4]), "=f"(d.x[5]),
     "=f"(d.x[2]), "=f"(d.x[3]), "=f"(d.x[6]), "=f"(d.x[7])

    : "r"(a.x[0]),  "r"(a.x[1]),  "r"(a.x[2]),  "r"(a.x[3]),
      "r"(b.x[0]),  "r"(b.x[2]),  "r"(b.x[1]),  "r"(b.x[3]),

      "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[4]),  "f"(c.x[5]),
      "f"(c.x[2]),  "f"(c.x[3]),  "f"(c.x[6]),  "f"(c.x[7])
    );
}


DEVICE_INLINE void mma_sync_sparse(
    fragment_c_sparse<ShapeBase<16, 8, 16>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 16>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const uint &e, const int psel)
{
    if (psel==0)
        asm volatile(
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10,%11}, %12, 0x0;\n"
        :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(a.x[0]),  "r"(a.x[1]),
          "r"(b.x[0]),  "r"(b.x[1]),
          "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e)
        );
    else if (psel==1)
        asm volatile(
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10,%11}, %12, 0x1;\n"
        :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(a.x[0]),  "r"(a.x[1]),
          "r"(b.x[0]),  "r"(b.x[1]),
          "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e)
        );
    else if (psel==2)
        asm volatile(
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10,%11}, %12, 0x2;\n"
        :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(a.x[0]),  "r"(a.x[1]),
          "r"(b.x[0]),  "r"(b.x[1]),
          "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e)
        );
    else
        asm volatile(
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10,%11}, %12, 0x3;\n"
        :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(a.x[0]),  "r"(a.x[1]),
          "r"(b.x[0]),  "r"(b.x[1]),
          "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e)
        );
}

DEVICE_INLINE void mma_sync_sparse_v(
    fragment_c_sparse<ShapeBase<16, 8, 16>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 16>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 16>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const fragment_meta_sparse<ShapeBase<16, 8, 16>> &e)
{
    /* asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10,%11}, %12, 0x0;\n"
    :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
    : "r"(a.x[0+offset]),  "r"(a.x[offset+1]),
        "r"(b.x[0]),  "r"(b.x[1]),
        "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e)
    ); */
}

DEVICE_INLINE void mma_sync_sparse_v(
    fragment_c_sparse<ShapeBase<16, 8, 32>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 32>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const fragment_meta_sparse<ShapeBase<16, 8, 32>> &e)
{
    /* asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, "
        "{%10,%11}, %12, 0x0;\n"
    :"=r"(d.x[0]), "=r"(d.x[1])
    : "r"(a.x[0]),  "r"(a.x[1]),  "r"(a.x[4]),  "r"(a.x[5]),
      "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
      "r"(c.x[0]),  "r"(c.x[1]),  "r"(e.x[0])
    );


    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7, %8, %9}, "
        "{%10,%11}, %12, 0x0;\n"
    :"=r"(d.x[2]), "=r"(d.x[3])
    : "r"(a.x[2]),  "r"(a.x[3]),  "r"(a.x[6]),  "r"(a.x[7]),
      "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
      "r"(c.x[2]),  "r"(c.x[3]),  "r"(e.x[1])
    ); */
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x0;\n"
    :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
    : "r"(a.x[0]),  "r"(a.x[1]),  "r"(a.x[4]),  "r"(a.x[5]),
        "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
        "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e.x[0])
    );

    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x0;\n"
    :"=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])
    : "r"(a.x[2]),  "r"(a.x[3]),  "r"(a.x[6]),  "r"(a.x[7]),
        "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
        "f"(c.x[4]),  "f"(c.x[5]),  "f"(c.x[6]),  "f"(c.x[7]),  "r"(e.x[1])
    );

}

DEVICE_INLINE void mma_sync_sparse(
    fragment_c_sparse<ShapeBase<16, 8, 32>, float> &d,
    const fragment_a_sparse_cm<ShapeBase<16, 8, 32>> &a,
    const fragment_b_sparse_rm<ShapeBase<16, 8, 32>> &b,
    const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const uint &e, const int psel)
{
    /* if (psel==0)
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
            "{%12,%13,%14,%15}, %16, 0x0;\n"
        :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(a.x[0]),  "r"(a.x[1]),  "r"(a.x[2]),  "r"(a.x[3]),
          "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
          "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e)
        );
    else
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
            "{%12,%13,%14,%15}, %16, 0x1;\n"
        :"=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
        : "r"(a.x[0]),  "r"(a.x[1]),  "r"(a.x[2]),  "r"(a.x[3]),
          "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
          "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),  "r"(e)
        ); */
}
//#endif

template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<16, 16, 8 >, float> &c,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane / 4;
    int col  = (lane & 3) *2;
    int offset_ = offset + row*ldm + col;
    F f;
    *(half2*)(base + f(offset_))             = __float22half2_rn(*(float2*)&c.x[0]);
    *(half2*)(base + f(offset_ + 8*ldm))     = __float22half2_rn(*(float2*)&c.x[2]);
    *(half2*)(base + f(offset_ + 8))         = __float22half2_rn(*(float2*)&c.x[4]);
    *(half2*)(base + f(offset_ + 8 + 8*ldm)) = __float22half2_rn(*(float2*)&c.x[6]);
}

//#if  __CUDA_ARCH__ >= 800
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<16, 16, 16>, float> &c,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane / 4;
    int col  = (lane & 3) *2;
    int offset_ = offset + row*ldm + col;
    F f;
    *(half2*)(base + f(offset_))             = __float22half2_rn(*(float2*)&c.x[0]);
    *(half2*)(base + f(offset_ + 8*ldm))     = __float22half2_rn(*(float2*)&c.x[2]);
    *(half2*)(base + f(offset_ + 8))         = __float22half2_rn(*(float2*)&c.x[4]);
    *(half2*)(base + f(offset_ + 8 + 8*ldm)) = __float22half2_rn(*(float2*)&c.x[6]);
}

template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const half *base, const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row  = lane / 4;
    int col  = (lane & 3) *2;
    int offset_ = offset + row*ldm + col;
    F f;
    *(half2*)(base + f(offset_))             = __float22half2_rn(*(float2*)&c.x[0]);
    *(half2*)(base + f(offset_ + 8*ldm))     = __float22half2_rn(*(float2*)&c.x[2]);
}

template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const half *base, const int offset, const int ldm, const int brow)
{
    int lane = threadIdx.x & 31;
    int row  = lane / 4;
    int col  = (lane & 3) *2;
    int offset_ = offset + row*ldm*brow + col;
    F f;
    *(half2*)(base + f(offset_))             = __float22half2_rn(*(float2*)&c.x[0]);
    *(half2*)(base + f(offset_ + ldm))     = __float22half2_rn(*(float2*)&c.x[2]);
}

template<class F> DEVICE_INLINE
void store_matrix_sync_128(const fragment_c_sparse<ShapeBase<16, 8, 16>, float> &c,
    const half *base, const int offset, const int ldm)
{
    F f;
}

typedef struct __align__(16) ehalf8 {
    __device__ __forceinline__ ehalf8() {}
    __device__ __forceinline__ ehalf8(uint v) : x(v), y(v), z(v), w(v) {}
    __device__ __forceinline__ ehalf8(uint v0, uint v1, uint v2, uint v3) : x(v0), y(v1), z(v2), w(v3) {}
    uint x;
    uint y;
    uint z;
    uint w;
} ehalf8;

typedef struct __align__(16) float8 {
    __device__ __forceinline__ float8() { }
    __device__ __forceinline__ float8(float4 x, float4 y) : a(x), b(y) {}
    //__device__ __forceinline__ float8(const float val) : a({val,val,val,val}), b({val,val,val,val}) { }
    float4 a;
    float4 b;
} float8;

__device__ __forceinline__ ehalf8 to_ehalf(float8 v)
{
    ehalf8 r;
    asm("{\n\t"
        ".reg .f16 v<8>;\n\t"
        "cvt.rn.f16.f32 v0, %4 ;\n\t"
        "cvt.rn.f16.f32 v1, %5 ;\n\t"
        "cvt.rn.f16.f32 v2, %6 ;\n\t"
        "cvt.rn.f16.f32 v3, %7 ;\n\t"
        "cvt.rn.f16.f32 v4, %8 ;\n\t"
        "cvt.rn.f16.f32 v5, %9 ;\n\t"
        "cvt.rn.f16.f32 v6, %10;\n\t"
        "cvt.rn.f16.f32 v7, %11;\n\t"
        "mov.b32 %0, {v0, v1};\n\t"
        "mov.b32 %1, {v2, v3};\n\t"
        "mov.b32 %2, {v4, v5};\n\t"
        "mov.b32 %3, {v6, v7};\n\t"
        "}" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
            :  "f"(v.a.x),"f"(v.a.y),"f"(v.a.z),"f"(v.a.w),
               "f"(v.b.x),"f"(v.b.y),"f"(v.b.z),"f"(v.b.w));
    return r;
}

__device__ __forceinline__ uint to_half2(float2 v)
{
    uint ret;
    asm("{                     \n\t"
        ".reg .f16 a, b;       \n\t"
        "cvt.rn.f16.f32 a, %1; \n\t"
        "cvt.rn.f16.f32 b, %2; \n\t"
        "mov.b32 %0, {a, b};   \n\t"
        "}" : "=r"(ret) : "f"(v.x),"f"(v.y));
    return ret;
}

__device__ __forceinline__ uint2 to_half4(float4 v)
{
    uint2 r;
    asm("{\n\t"
        ".reg .f16 a, b, c, d; \n\t"
        "cvt.rn.f16.f32 a, %2; \n\t"
        "cvt.rn.f16.f32 b, %3; \n\t"
        "cvt.rn.f16.f32 c, %4; \n\t"
        "cvt.rn.f16.f32 d, %5; \n\t"
        "mov.b32 %0, {a, b};   \n\t"
        "mov.b32 %1, {c, d};   \n\t"
        "}" : "=r"(r.x), "=r"(r.y) : "f"(v.x),"f"(v.y), "f"(v.z),"f"(v.w));
    return r;
}

DEVICE_INLINE
int div16(int numerator, int magic, int shift)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, 0;" : "=r"(res) : "r"(numerator), "r"(magic));
    return res >> shift;
}
DEVICE_INLINE
int mod16(int numerator, int div, int maxdiv)
{
    int res;
    asm("vmad.s32.u32.u32 %0, -%1.h0, %2.h0, %3;" : "=r"(res) : "r"(div), "r"(maxdiv), "r"(numerator));
    return res;
}
DEVICE_INLINE
int mad16(int a, int b, int c)
{
    int res;
    asm("vmad.s32.u32.u32 %0, %1.h0, %2.h0, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const half *base, const int offset, const int ldm)
{
    F f;

    //bhalf8 src = to_bhalf(float8(*(float4*)&c.x[0], *(float4*)&c.x[4]));

    *(half2*)(base+f(offset))       = __float22half2_rn(*(float2*)&c.x[0]);
    *(half2*)(base+f(offset+ldm))   = __float22half2_rn(*(float2*)&c.x[2]);
    *(half2*)(base+f(offset+2*ldm)) = __float22half2_rn(*(float2*)&c.x[4]);
    *(half2*)(base+f(offset+3*ldm)) = __float22half2_rn(*(float2*)&c.x[6]);

    /* *(uint*)(base+f(offset))        = to_half2(*(float2*)&c.x[0]);
    *(uint*)(base+f(offset+ldm))    = to_half2(*(float2*)&c.x[2]);
    *(uint*)(base+f(offset+2*ldm))  = to_half2(*(float2*)&c.x[4]);
    *(uint*)(base+f(offset+3*ldm))  = to_half2(*(float2*)&c.x[6]); */

    /* ehalf8 src =  to_ehalf(float8(*(float4*)&c.x[0], *(float4*)&c.x[4]));
    *(uint*)(base+f(offset))       = src.x;
    *(uint*)(base+f(offset+ldm))   = src.y;
    *(uint*)(base+f(offset+2*ldm)) = src.z;
    *(uint*)(base+f(offset+3*ldm)) = src.w; */
}

template<class F> DEVICE_INLINE
void store_matrix_sync_128(const fragment_c_sparse<ShapeBase<16, 8, 32>, float> &c,
    const half *base, const int offset, const int ldm)
{

    uint2 src[2];
    src[0] = to_half4(*(float4*)&c.x[0]);
    src[1] = to_half4(*(float4*)&c.x[4]);

   *(float4*)(base+offset) = *(float4*)(&src[0]);
}

DEVICE_INLINE
void ld_shared_float4(const void *a, uint *r)
{
    unsigned smem_ptr = get_smem_ptr(a);
    asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"  : "=r"(r[0]),"=r"(r[4]),"=r"(r[8]),"=r"(r[12]) : "r"(smem_ptr));
}

template<class F> DEVICE_INLINE
void store_matrix_sync2(const half2 (&out)[4], const half *base, const int offset, const int ldm, const int brow)
{
    int lane = threadIdx.x & 31;
    int row  = lane / 4;
    int col  = (lane & 3) *ldm;
    int offset_ = offset + row*ldm*brow + col;
    F f;
    /* *(half2*)(base + f(offset_))             = __float22half2_rn(*(float2*)&c.x[0]);
    *(half2*)(base + f(offset_ + ldm))     = __float22half2_rn(*(float2*)&c.x[2]); */

    *((float4*)(base + f(offset_)))     = *((float4*)&out);
}
//#endif

// *** ldmatrix ***
template<bool trans, int num_reg, int nbit>
DEVICE_INLINE void ldmatrix(uint *dst, const void *src)
{
    // no f32 transpose is supported in current cuda
    static_assert((!trans) || nbit==16);

    unsigned smem_ptr = get_smem_ptr(src);

    uint* x = dst;

    if (!trans) {
        if (num_reg==4) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
            : "r"(smem_ptr));
        }
        else if (num_reg==2) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(x[0]), "=r"(x[1])
            : "r"(smem_ptr));
        }
        else if (num_reg==1) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
            : "=r"(x[0])
            : "r"(smem_ptr));
        }
        else assert(0);
    }
    else { // trans
        if (num_reg==4) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
            : "r"(smem_ptr));
        }
        else if (num_reg==2) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(x[0]), "=r"(x[1])
            : "r"(smem_ptr));
        }
        else if (num_reg==1) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
            : "=r"(x[0])
            : "r"(smem_ptr));
        }
        else assert(0);
    }
}

/* #else // __CUDA_ARCH__ >= 750

template<typename Shape> struct fragment_a_colmajor;
template<typename Shape> struct fragment_b_rowmajor;
template<typename Shape, typename Accumulator> struct fragment_c;

// *** 16x16x16 f32.f16 ***
template<> struct fragment_a_colmajor<ShapeBase<16, 16, 16>> { uint x[8]; };
template<> struct fragment_b_rowmajor<ShapeBase<16, 16, 16>> { uint x[8]; };
template<> struct fragment_c<ShapeBase<16, 16, 16>, float> { float x[8] = {0};};

// *** 32x32x16 f32.f16 ***
template<> struct fragment_a_colmajor<ShapeBase<32, 32, 16>> { uint x[16]; };
template<> struct fragment_b_rowmajor<ShapeBase<32, 32, 16>> { uint x[16]; };
template<> struct fragment_c<ShapeBase<32, 32, 16>, float> { float x[32] = {0};};

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<32, 32, 16>> &a,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<32, 32, 16>> &b,
    const half *base, const int offset, const int ldm);
DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 16>, float> &d,
    const fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const fragment_c<ShapeBase<16, 16, 16>, float> &c);
DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<32, 32, 16>, float> &d,
    const fragment_a_colmajor<ShapeBase<32, 32, 16>> &a,
    const fragment_b_rowmajor<ShapeBase<32, 32, 16>> &b,
    const fragment_c<ShapeBase<32, 32, 16>, float> &c);
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<16, 16, 16>, float> &c,
    const half *base, const int offset, const int ldm);
template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<32, 32, 16>, float> &c,
    const half *base, const int offset, const int ldm);

// *** implementation ***
template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const half* base, const int offset, const int ldm)
{
    F f;
    int lane = threadIdx.x & 31;
    int row = lane & 3;
    int col = ((lane & 4)<<1) + ((lane & 16) >> 2);
    int frag_offset = offset + row * ldm + col;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const half *src = base + f(frag_offset + 4*k*ldm);
        *(uint2*)&a.x[k*2] = *(uint2*)src;
    }
}

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_a_colmajor<ShapeBase<32, 32, 16>> &a,
    const half* base, const int offset, const int ldm)
{
    F f;
    int lane = threadIdx.x & 31;
    int row = lane & 3;
    int col = ((lane & 4)<<2) + ((lane & 16) >> 1);
    int frag_offset = offset + row * ldm + col;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const half *src = base + f(frag_offset + 4*k*ldm);
        *(uint4*)&a.x[k*4] = *(uint4*)src;
    }
}


template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const half* base, const int offset, const int ldm
) {
    F f;
    int lane_rank = threadIdx.x & 31;
    int row = lane_rank & 3;
    int col = (lane_rank & 8) + ((lane_rank & 16) >> 2);
    int frag_offset = offset + row * ldm + col;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const half *src = base + f(frag_offset + 4*k*ldm);
        *(uint2*)&b.x[k*2] = *(uint2*)src;
    }
}

template<class F> DEVICE_INLINE
void load_matrix_sync(fragment_b_rowmajor<ShapeBase<32, 32, 16>> &b,
    const half* base, const int offset, const int ldm
) {
    F f;
    int lane_rank = threadIdx.x & 31;
    int row = lane_rank & 3;
    int col = ((lane_rank & 8)<<1) + ((lane_rank & 16) >> 1);
    int frag_offset = offset + row * ldm + col;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const half *src = base + f(frag_offset + 4*k*ldm);
        *(uint4*)&b.x[k*4] = *(uint4*)src;
    }
}

DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<16, 16, 16>, float> &d,
    const fragment_a_colmajor<ShapeBase<16, 16, 16>> &a,
    const fragment_b_rowmajor<ShapeBase<16, 16, 16>> &b,
    const fragment_c<ShapeBase<16, 16, 16>, float> &c
) {
    asm volatile ("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%8, %9 }, {%16,%17}, "
    "{%24,%25,%26,%27,%28,%29,%30,%31};"
    "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%10,%11}, {%18,%19}, "
    "{%0, %1, %2, %3, %4, %5, %6, %7 };"
    "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%12,%13}, {%20,%21}, "
    "{%0, %1, %2, %3, %4, %5, %6, %7 };"
    "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%14,%15}, {%22,%23}, "
    "{%0, %1, %2, %3, %4, %5, %6, %7 };"
    : "=f"(d.x[0]),  "=f"(d.x[1]),  "=f"(d.x[2]),  "=f"(d.x[3]),
      "=f"(d.x[4]),  "=f"(d.x[5]),  "=f"(d.x[6]),  "=f"(d.x[7])
    : "r"(a.x[0]),  "r"(a.x[1]),  "r"(a.x[2]),  "r"(a.x[3]),
      "r"(a.x[4]),  "r"(a.x[5]),  "r"(a.x[6]),  "r"(a.x[7]),
      "r"(b.x[0]),  "r"(b.x[1]),  "r"(b.x[2]),  "r"(b.x[3]),
      "r"(b.x[4]),  "r"(b.x[5]),  "r"(b.x[6]),  "r"(b.x[7]),
      "f"(c.x[0]),  "f"(c.x[1]),  "f"(c.x[2]),  "f"(c.x[3]),
      "f"(c.x[4]),  "f"(c.x[5]),  "f"(c.x[6]),  "f"(c.x[7]));
}

DEVICE_INLINE void mma_sync(fragment_c<ShapeBase<32, 32, 16>, float> &d,
    const fragment_a_colmajor<ShapeBase<32, 32, 16>> &a,
    const fragment_b_rowmajor<ShapeBase<32, 32, 16>> &b,
    const fragment_c<ShapeBase<32, 32, 16>, float> &c
) {
    #pragma unroll
    for (int mm = 0; mm < 2; mm++) {
        #pragma unroll
        for (int nn = 0; nn < 2; nn++) {
            asm  volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%8, %9 }, {%16,%17}, "
            "{%24,%25,%26,%27,%28,%29,%30,%31};"
            "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%10,%11}, {%18,%19}, "
            "{%0, %1, %2, %3, %4, %5, %6, %7 };"
            "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%12,%13}, {%20,%21}, "
            "{%0, %1, %2, %3, %4, %5, %6, %7 };"
            "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7 }, {%14,%15}, {%22,%23}, "
            "{%0, %1, %2, %3, %4, %5, %6, %7 };"
            : "=f"(d.x[8*(mm*2+nn)+0]),  "=f"(d.x[8*(mm*2+nn)+1]),
              "=f"(d.x[8*(mm*2+nn)+2]),  "=f"(d.x[8*(mm*2+nn)+3]),
              "=f"(d.x[8*(mm*2+nn)+4]),  "=f"(d.x[8*(mm*2+nn)+5]),
              "=f"(d.x[8*(mm*2+nn)+6]),  "=f"(d.x[8*(mm*2+nn)+7])
            : "r"(a.x[2*mm+0]),  "r"(a.x[2*mm+1]),  "r"(a.x[2*mm+4]),  "r"(a.x[2*mm+5]),
              "r"(a.x[2*mm+8]),  "r"(a.x[2*mm+9]),  "r"(a.x[2*mm+12]), "r"(a.x[2*mm+13]),
              "r"(b.x[2*nn+0]),  "r"(b.x[2*nn+1]),  "r"(b.x[2*nn+4]),  "r"(b.x[2*nn+5]),
              "r"(b.x[2*nn+8]),  "r"(b.x[2*nn+9]),  "r"(b.x[2*nn+12]), "r"(b.x[2*nn+13]),
              "f"(c.x[8*(mm*2+nn)+0]),  "f"(c.x[8*(mm*2+nn)+1]),
              "f"(c.x[8*(mm*2+nn)+2]),  "f"(c.x[8*(mm*2+nn)+3]),
              "f"(c.x[8*(mm*2+nn)+4]),  "f"(c.x[8*(mm*2+nn)+5]),
              "f"(c.x[8*(mm*2+nn)+6]),  "f"(c.x[8*(mm*2+nn)+7]));
        }
    }
}

template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<16, 16, 16>, float> &c,
    const half *base, const int offset, const int ldm)
{
    F f;
    int lane = threadIdx.x & 31;
    int row    = ((lane & 4)<<1) + ((lane & 16) >> 2) + (lane & 1);
    int col    = (lane & 8) + (lane & 2);
    int _offset= offset + row * ldm + col;

    *(__half2*)(base + f(_offset))             = __float22half2_rn( *(float2*)&c.x[0] );
    *(__half2*)(base + f(_offset + 2*ldm))     = __float22half2_rn( *(float2*)&c.x[2] );
    *(__half2*)(base + f(_offset + 4))         = __float22half2_rn( *(float2*)&c.x[4] );
    *(__half2*)(base + f(_offset + 2*ldm + 4)) = __float22half2_rn( *(float2*)&c.x[6] );
}

template<class F> DEVICE_INLINE
void store_matrix_sync(const fragment_c<ShapeBase<32, 32, 16>, float> &c,
    const half *base, const int offset, const int ldm)
{
    F f;
    int lane = threadIdx.x & 31;
    int row    = ((lane & 4)<<2) + ((lane & 16) >> 1) + (lane & 1);
    int col    = ((lane & 8)<<1) + (lane & 2);
    int _offset= offset + row * ldm + col;

    #pragma unroll
    for (int mm = 0; mm < 2; mm++) {
    #pragma unroll
    for (int nn = 0; nn < 2; nn++) {
    #pragma unroll
    for (int nx = 0; nx < 2; nx++) {
    #pragma unroll
    for (int mx = 0; mx < 2; mx++) {
        *(__half2*)(base + f(_offset + (2*mx + 4*mm)*ldm + (8*nx + 4*nn))) =
            __float22half2_rn(*(float2*)&c.x[2*(mx + nx*2 + nn*4 + mm*8)]);
    }
    }
    }
    }
}

#endif // __CUDA_ARCH__ >= 750 */

} // namespace my_mma
}
