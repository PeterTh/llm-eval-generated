cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The single source file contains CUDA kernels; compile it as CUDA
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set_target_properties(cholesky PROPERTIES CUDA_ARCHITECTURES native)
endif()

# Enable optimizations (host flags forwarded through nvcc)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link with the host C++ toolchain so OpenMP/MPI link flags are handled natively
set_target_properties(cholesky PROPERTIES LINKER_LANGUAGE CXX)

# Link math library, MPI, OpenMP and the CUDA runtime
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
