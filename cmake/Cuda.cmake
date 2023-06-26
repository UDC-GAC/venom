function(cuda_find_library out_path lib_name)
  find_library(${out_path} ${lib_name} PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    PATH_SUFFIXES lib lib64 REQUIRED)
endfunction()

function(create_cuda_gencode_flags out archs_args)
  set(archs ${archs_args} ${ARGN})
  set(tmp "")
  foreach(arch IN LISTS archs)
    set(tmp "${tmp} -gencode arch=compute_${arch},code=sm_${arch} -cubin -Xlinker=--whole-archive    \
    -Xlinker=--no-whole-archiv")
  endforeach(arch)
  set(${out} ${tmp} PARENT_SCOPE)
endfunction()
