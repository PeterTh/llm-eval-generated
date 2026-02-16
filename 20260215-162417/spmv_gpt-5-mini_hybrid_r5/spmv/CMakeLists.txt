cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language for compiling this .cpp as CUDA source
enable_language(CUDA)

add_executable(spmv spmv.cpp)
# Treat the .cpp as CUDA to allow __global__ kernels in the same file
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Find MPI, OpenMP and CUDA runtime
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Enable optimizations
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3> $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link required libraries
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# Ensure position independent code is off for best performance on accelerators
set_target_properties(spmv PROPERTIES POSITION_INDEPENDENT_CODE OFF)
