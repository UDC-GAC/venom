set(SPMM_BENCHS_SOURCES
	benchmark_spmm.cpp)

#cmake_policy(SET CMP0104 OLD)

set_source_files_properties(${SPMM_BENCHS_SOURCES}  PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${SDDMM_BENCHS_SOURCES} PROPERTIES LANGUAGE CUDA)

set(SPMM_TESTS_SOURCES
	spmm_test.cpp)

set_source_files_properties(${SPMM_TESTS_SOURCES} PROPERTIES LANGUAGE CUDA)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Type of build, options are: Debug, Release" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g2 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_subdirectory(format)
add_subdirectory(dataset)
add_subdirectory(gemm)
add_subdirectory(spmm)

# cusparselt_example check-env utilities
function(set_with_fallback VARIABLE FALLBACK)
  if (NOT DEFINED ${VARIABLE} OR ${VARIABLE} STREQUAL "")
    set(${VARIABLE} $ENV{${VARIABLE}} CACHE INTERNAL ${VARIABLE})
    if (${VARIABLE} STREQUAL "")
      if (NOT ${FALLBACK} STREQUAL "")
        set(${VARIABLE} $ENV{${FALLBACK}} CACHE INTERNAL ${VARIABLE})
      endif ()
    endif ()
  endif ()
endfunction()

set_with_fallback(CUSPARSELT_PATH CUSPARSELT_DIR)
if (CUSPARSELT_PATH STREQUAL "")
    message(FATAL_ERROR "Please set the environment variables CUSPARSELT_DIR to the path of the cuSPARSELt installation.")
endif ()

message(STATUS "Using CUSPARSELT_PATH = ${CUSPARSELT_DIR}")

set_with_fallback(CUDA_TOOLKIT_PATH CUDA_TOOLKIT_DIR)
if (CUSPARSELT_PATH STREQUAL "")
	message(FATAL_ERROR "Please set the environment variables CUDA_TOOLKIT_DIR to the path of the toolkit installation.")
endif ()

add_executable(benchmark_spmm
	${SPMM_BENCHS_SOURCES})
set_source_files_properties(${benchmark_spmm} PROPERTIES LANGUAGE CUDA)


target_include_directories(spmm
    PUBLIC ${CUSPARSELT_PATH}/include
)

target_link_directories(spmm
    PUBLIC ${CUSPARSELT_PATH}/lib
    PUBLIC ${CUDA_TOOLKIT_PATH}/lib64
)

target_link_libraries(spmm
PUBLIC libcusparseLt.so
)

target_link_libraries(benchmark_spmm
	formats datasets gemm spmm boost_system boost_filesystem sputnik spmm_ve_amp spatha spatha_util ${SPUTNIK_BENCHMARK_LIBS} cusparse cublas cusparseLt dl libcusparseLt.so cublasLt cuda)