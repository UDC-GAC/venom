set(GEMM_SOURCES
	gemm.hpp
	gemm.cpp
	cublas_gemm.hpp
    cublas_gemm.cpp
	cublasLt_gemm.hpp
    cublasLt_gemm.cpp
)

set_source_files_properties(${GEMM_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(gemm SHARED
	${GEMM_SOURCES})

target_link_directories(gemm
	PUBLIC ${CUDA_TOOLKIT_PATH}/lib64
)

target_include_directories(gemm
	PUBLIC .)

