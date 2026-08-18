cmake_minimum_required(VERSION 3.24)

# CUDA 12 rejects GCC 13 before CMake gets far enough to configure flags.
# Select its supported GCC 12 host compiler when it is available, while
# leaving other CUDA releases and explicitly selected toolchains untouched.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    execute_process(COMMAND nvcc --version OUTPUT_VARIABLE NVCC_VERSION_TEXT
                    ERROR_QUIET)
    if(NVCC_VERSION_TEXT MATCHES "release 12\\.")
        find_program(CUDA12_HOST_COMPILER NAMES g++-12)
        if(CUDA12_HOST_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER "${CUDA12_HOST_COMPILER}" CACHE FILEPATH
                "Host compiler supported by CUDA 12")
        endif()
    endif()
endif()

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the original source and executable names; compile the existing source
# as CUDA so host MPI/OpenMP and device kernels remain in one translation unit.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET cholesky PROPERTY CUDA_ARCHITECTURES native)
endif()
set_property(TARGET cholesky PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)
