// Lots of small util functions and definitions.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>
#include <string>

namespace spatha {
// *** math utilities ***

#define CEIL(x, y) (((x) + (y) -1)/(y))
#define ROUND_UP(x, y) ((CEIL((x), (y)))*(y))

// macro to declare a device-side function
#define DEVICE_INLINE __device__ __forceinline__

// *** type for storing a 3D shape  ***
template<int M_, int N_, int K_> struct ShapeBase {
    static constexpr int M = M_, N = N_, K = K_;
};

}
