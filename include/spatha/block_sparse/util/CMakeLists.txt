set(SPMM_OPENSPARSELT_SOURCES
    ./random_mask.h
    ./random_mask.cc
)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-DGPU_CC=${CUDA_ARCHS})
add_library(spatha_util STATIC
	${SPMM_OPENSPARSELT_SOURCES})

target_include_directories(spatha_util
	PUBLIC .)