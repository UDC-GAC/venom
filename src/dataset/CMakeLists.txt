set(DATASET_SOURCES
	dataset.hpp
	dataset.cpp)

set_source_files_properties(${DATASET_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(datasets STATIC
	${DATASET_SOURCES})

target_include_directories(datasets
	PUBLIC .)

