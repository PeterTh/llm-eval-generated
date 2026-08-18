cmake_minimum_required(VERSION 3.24)

# CUDA 12.0 supports GCC 12 as a host compiler.  Prefer it when present so the
# project also configures on distributions whose default GCC is newer.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(CUDA_COMPATIBLE_HOST_COMPILER NAMES g++-12 g++-11 g++)
    if(CUDA_COMPATIBLE_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${CUDA_COMPATIBLE_HOST_COMPILER}" CACHE FILEPATH
            "Host compiler used by NVCC")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
        "CUDA architectures (defaults to the installed GPU)")
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=true;--extra-device-vectorization;
      -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

target_link_libraries(unstructured PRIVATE m)
