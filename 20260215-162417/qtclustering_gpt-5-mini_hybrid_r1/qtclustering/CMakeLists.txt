cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language and compile the single source as CUDA to allow __global__ kernels in the .cpp
enable_language(CUDA)

add_executable(qtclustering qtclustering.cpp)

# Compile qtclustering.cpp with CUDA language
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Optimization flags for C++ and CUDA; enable OpenMP for host code and pass to NVCC
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
)

# Link against MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
