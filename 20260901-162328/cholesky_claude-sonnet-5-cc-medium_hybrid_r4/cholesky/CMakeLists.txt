cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# The source contains CUDA kernels (__global__), so compile it with nvcc.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math
        -Xcompiler=-O3,-march=native,-Wall,-Wextra,${OpenMP_CXX_FLAGS}>
)

target_include_directories(cholesky PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(cholesky PRIVATE MPI::MPI_C OpenMP::OpenMP_CXX m)
