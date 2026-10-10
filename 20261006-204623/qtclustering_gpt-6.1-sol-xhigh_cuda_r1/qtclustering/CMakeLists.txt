cmake_minimum_required(VERSION 3.24)

# CUDA 12 supports GCC through version 12. Honor explicit host compiler choices.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(QT_CUDA_HOST_COMPILER NAMES g++-12 g++-11)
    if(QT_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${QT_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler")
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the original source filename and executable name.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)
target_link_libraries(qtclustering PRIVATE m)
