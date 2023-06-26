include(cmake/Cuda.cmake)

cuda_find_library(CUDART_LIBRARY cudart_static)
cuda_find_library(CUSPARSE_LIBRARY cusparse_static)
set(CUBLASLT_LIBRARY cublasLt)
list(APPEND SpMM_CNN_LIBS "cudart_static;cusparse_static;cublas_static")
#list(APPEND SpMM_CNN_LIBS "cudart_static;cublas_static")

function(create_cuda_gencode_flags out archs_args)
  set(archs ${archs_args} ${ARGN})
  set(tmp "")
  foreach(arch IN LISTS archs)
    set(tmp "${tmp} -m64 -lineinfo -gencode arch=compute_${arch},code=sm_${arch} -Xlinker=--whole-archive    \
    -I ${CUSPARSELT_DIR}/include \
    -Xlinker=${CUSPARSELT_DIR}/lib/libcusparseLt_static.a      \
    -Xlinker=--no-whole-archiv")
  endforeach(arch)
  set(${out} ${tmp} PARENT_SCOPE)
endfunction()