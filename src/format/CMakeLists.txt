set(FORMAT_SOURCES
	format.cpp
	format.hpp
	format_cxx.hpp
	format_cxx.cpp
	format_csr.cpp
	format_csr.hpp
	format_cvs.cpp
	format_cvs.hpp
	format_sr_nm.hpp
	format_sr_nm.cpp)

set_source_files_properties(${FORMAT_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(formats STATIC
	${FORMAT_SOURCES})