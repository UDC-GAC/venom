#pragma once
#include "base.h"    // DEVICE_INLINE

namespace spatha {

template<int N> struct HalfVector;
template<> struct HalfVector<8> {
    __half2 x[4];

    DEVICE_INLINE
    void ld(const half *src) {
        *(float4*)x = *(float4*)src;
    }

    DEVICE_INLINE
    void st(half *dst) {
        *(float4*)dst = *(float4*)x;
    }

    DEVICE_INLINE
    void mul(float &a) {
        __half2 aa = __float2half2_rn(a);
        for (int i = 0; i < 4; i++)
            x[i] = __hmul2(x[i], aa);
    }

    DEVICE_INLINE
    void hfma(float &b, HalfVector<8> &c) {
    // x = x*b+c
        __half2 bb = __float2half2_rn(b);
        for (int i = 0; i < 4; i++) {
            x[i] = __hfma2(x[i], bb, c.x[i]);
        }
    }

    // DEVICE_INLINE
    // void hfma_relu(float &b, float &c) {
    // // x = x*b+c
    //     __half2 bb = __float2half2_rn(b);
    //     __half2 cc = __float2half2_rn(c);
    //     for (int i = 0; i < 4; i++) {
    //         x[i] = __hfma2_relu(x[i], bb, cc);
    //     }
    // }
};

}