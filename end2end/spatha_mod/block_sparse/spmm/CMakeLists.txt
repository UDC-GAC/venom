set(SPMM_SPATHA_SOURCES
    ./blockwise_library.cu
)

#cmake_policy(SET CMP0104 OLD)

set_source_files_properties(${SPMM_SPATHA_SOURCES} PROPERTIES LANGUAGE CUDA)

include(GNUInstallDirs)

add_definitions(-DGPU_CC=${CUDA_ARCHS})

add_library(spatha
	STATIC ${SPMM_SPATHA_SOURCES})

target_include_directories(spatha
	PRIVATE ${PYTHON_INCLUDE_DIRS}
)

message(STATUS "TORCH LIBRARIES ENV VARIABLE: ${TORCH_LIBRARIES}")
message(STATUS "TORCH DIRS ENV VARIABLE: ${TORCH_INCLUDE_DIRS}")

target_link_libraries(spatha
	PUBLIC
	Python::Python
	${CMAKE_TORCH_LIBRARIES}
)

target_include_directories(spatha
	PUBLIC ${CMAKE_TORCH_DIRS})
