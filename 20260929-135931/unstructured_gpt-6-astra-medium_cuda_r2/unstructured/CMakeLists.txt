cmake_minimum_required(VERSION 3.24)

# CUDA 12.0 supports GCC through version 12. Respect an explicitly selected
# host compiler, otherwise prefer a compatible installed GCC.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(CUDA_COMPATIBLE_HOST NAMES g++-12 g++-11)
    if(CUDA_COMPATIBLE_HOST)
        set(CMAKE_CUDA_HOST_COMPILER "${CUDA_COMPATIBLE_HOST}" CACHE FILEPATH "CUDA host compiler")
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

# Keep the existing source filename and executable name.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
# Disable contraction to preserve the reference's separate double operations.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
target_link_libraries(unstructured PRIVATE m)
