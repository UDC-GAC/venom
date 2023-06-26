set(SPMM_OPENSPARSELT_SOURCES
    ./blockwise_library.cu
)

#cmake_policy(SET CMP0104 OLD)

set_source_files_properties(${SPMM_OPENSPARSELT_SOURCES} PROPERTIES LANGUAGE CUDA)

include(GNUInstallDirs)

add_definitions(-DGPU_CC=${CUDA_ARCHS})

add_library(spatha
	STATIC ${SPMM_OPENSPARSELT_SOURCES})
