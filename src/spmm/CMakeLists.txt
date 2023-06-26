set(SPMM_SOURCES
	spmm.hpp
	spmm.cpp
	sputnik_spmm.hpp
	sputnik_spmm.cpp
	clasp_spmm.hpp
	clasp_spmm.cpp
	spatha_spmm.hpp
	spatha_spmm.cpp
	cusparseLt_spmm.hpp
  cusparseLt_spmm.cpp
)

set_source_files_properties(${SPMM_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(spmm STATIC
	${SPMM_SOURCES})

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


target_link_libraries(spmm
    PUBLIC spatha spatha_util cusparse cublas nvToolsExt cusparseLt dl cusparseLt )


target_link_directories(spmm
    PUBLIC ${CUSPARSELT_PATH}/lib
    PUBLIC ${CUDA_TOOLKIT_PATH}/lib64
)

target_link_libraries(spmm
	#PUBLIC cudart
    PUBLIC cusparse
    PUBLIC libcusparseLt.so
)
