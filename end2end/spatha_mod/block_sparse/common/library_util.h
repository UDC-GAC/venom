#pragma once

namespace spatha {

#define NAME_FUNC(type, fn, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE)\
type##_##BM##x##BLOCK_N##x##BLOCK_K##_##WARP_M##x##WARP_N##x##WARP_K##_##MMA_M##x##MMA_N##x##MMA_K##_##NSTAGE##_##fn##Fn


#define DECL_FUNC(type, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
extern type##InitFn_t NAME_FUNC(type, Init, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE); \
extern type##ExecFn_t NAME_FUNC(type, Exec, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE);

}