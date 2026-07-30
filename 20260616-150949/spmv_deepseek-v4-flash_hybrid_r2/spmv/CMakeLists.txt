cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Compile spmv.cpp with CUDA (nvcc handles .cpp as CUDA + host)
# Use CUDA language so .cpp file is compiled by nvcc, which forwards
# host-code to g++. The -Xcompiler flag ensures OpenMP pragmas reach g++.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Per-language compile options:
#   CUDA  → pass -fopenmp to the host compiler via -Xcompiler
#   CXX   → pass -fopenmp directly (fallback, not used since file is CUDA)
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

# Enable optimizations (CXX compile options also use -Xcompiler for CUDA)
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Set CUDA architecture (sm_70 = Volta; adjust for your GPU)
set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES "70")

# Link MPI and OpenMP
target_link_libraries(spmv MPI::MPI_CXX OpenMP::OpenMP_CXX)
