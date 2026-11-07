cmake_minimum_required(VERSION 3.18)

# nvcc 12.x only supports host compilers up to GCC 12; prefer a supported one
# if the default is newer and the user has not chosen a host compiler.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(_CUDA_HOST_GXX NAMES g++-12 g++-11 g++-10)
    if(_CUDA_HOST_GXX)
        set(CMAKE_CUDA_HOST_COMPILER "${_CUDA_HOST_GXX}" CACHE FILEPATH "CUDA host compiler")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# The source is CUDA code kept in a .cpp file; compile it with nvcc.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(unstructured m)
