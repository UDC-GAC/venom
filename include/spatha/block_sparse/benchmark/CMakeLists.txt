set(SPMM_SPATHA_BENCH_SOURCES
    ./argparse_util.h
    ./timing_util.h
    ./benchmark.spmm_nm.cu
)

set_source_files_properties(${SPMM_SPATHA_BENCH_SOURCES} PROPERTIES LANGUAGE CUDA)

#add_library(spmm SHARED
#	${SPMM_SOURCES})

add_definitions(-DGPU_CC=${CUDA_ARCHS})
add_library(spatha_bench STATIC
	${SPMM_SPATHA_BENCH_SOURCES})

target_link_libraries(spatha_bench
    PUBLIC cudart nvToolsExt
)

target_include_directories(spatha_bench
	PUBLIC .)