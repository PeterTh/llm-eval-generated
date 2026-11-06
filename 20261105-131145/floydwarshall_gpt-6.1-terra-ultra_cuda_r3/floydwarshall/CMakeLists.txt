cmake_minimum_required(VERSION 3.18)
# Preserve architecture choices supplied by the caller or CUDAARCHS.
if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
    set(FLOYDWARSHALL_ARCHITECTURES_SET ON)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

# Prefer the local GPU when supported; GPU-less build hosts retain the
# compiler default and callers can always set CMAKE_CUDA_ARCHITECTURES.
if(NOT FLOYDWARSHALL_ARCHITECTURES_SET AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5
   AND NOT "$ENV{CUDA_VISIBLE_DEVICES}" STREQUAL "-1")
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(COMMAND "${NVIDIA_SMI}" -L
                        RESULT_VARIABLE NVIDIA_SMI_RESULT
                        OUTPUT_VARIABLE NVIDIA_SMI_OUTPUT ERROR_QUIET)
        if(NVIDIA_SMI_RESULT EQUAL 0 AND NVIDIA_SMI_OUTPUT MATCHES "GPU [0-9]+:")
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The existing source file contains both the host program and CUDA kernels.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)
