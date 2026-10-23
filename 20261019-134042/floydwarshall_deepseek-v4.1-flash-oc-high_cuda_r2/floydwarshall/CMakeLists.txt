cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _cuda_caps
        RESULT_VARIABLE _cuda_caps_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_cuda_caps_result EQUAL 0 AND _cuda_caps)
        string(REPLACE "." "" _cuda_arch "${_cuda_caps}")
        string(REGEX REPLACE "[^0-9].*" "" _cuda_arch "${_cuda_arch}")
        set(CMAKE_CUDA_ARCHITECTURES "${_cuda_arch}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
endif()

add_executable(floydwarshall floydwarshall.cpp)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3>)

# Link math library
target_link_libraries(floydwarshall m)
