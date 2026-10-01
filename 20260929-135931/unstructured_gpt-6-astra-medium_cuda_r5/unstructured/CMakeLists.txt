cmake_minimum_required(VERSION 3.24)
# CUDA 12.0 requires a host compiler older than the system GCC 13 on some
# installations. Honor explicit CMake/environment choices first.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND "$ENV{CUDAHOSTCXX}" STREQUAL "")
    find_program(CUDA_COMPATIBLE_HOST NAMES g++-12 g++-11)
    if(CUDA_COMPATIBLE_HOST)
        set(CMAKE_CUDA_HOST_COMPILER "${CUDA_COMPATIBLE_HOST}" CACHE FILEPATH "CUDA host compiler")
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

# Compile the existing source as CUDA without renaming the file or executable.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

# Separate multiply/add operations preserve the reference result bit patterns.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
target_link_libraries(unstructured PRIVATE m)
