cmake_minimum_required(VERSION 3.24)

# CUDA 12.0 needs a host compiler older than GCC 13. Honor explicit compiler
# selections, and otherwise prefer a supported installed GCC.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(QT_CUDA_HOST_COMPILER NAMES g++-12 g++-11 g++)
    if(QT_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${QT_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)
set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(qtclustering m)
