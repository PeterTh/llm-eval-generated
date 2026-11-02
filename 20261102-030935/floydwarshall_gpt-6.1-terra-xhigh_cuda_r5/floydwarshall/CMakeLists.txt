cmake_minimum_required(VERSION 3.18)
# Prefer the build machine's GPU when it is visible; callers can always set
# CMAKE_CUDA_ARCHITECTURES explicitly for a different deployment target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS}
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(COMMAND "${NVIDIA_SMI}" -L
                        RESULT_VARIABLE GPU_QUERY_STATUS OUTPUT_VARIABLE GPU_LIST
                        ERROR_QUIET)
        if(GPU_QUERY_STATUS EQUAL 0 AND GPU_LIST MATCHES "GPU [0-9]+:")
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
    endif()
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
