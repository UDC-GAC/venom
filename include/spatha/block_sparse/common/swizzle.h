#pragma once
#include "base.h"

namespace spatha {

struct SwizzleIdentity {
    DEVICE_INLINE
    int operator()(int offset) {
        return offset;
    }
};

struct Swizzle8BWiseXor {
    DEVICE_INLINE
    int operator()(int offset) {
        return (offset ^
                ((offset & (7<<6))>>3));
    }
};

}