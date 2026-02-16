cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Ensure CUDA language is enabled
enable_language(CUDA)

add_executable(black_scholes black_scholes.cpp)

# Tell CMake to compile the single source file with NVCC (CUDA)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Optimization and warning flags
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
# CUDA specific flags
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -Xcompiler=-fopenmp>)

# Link MPI and OpenMP
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m cuda)

# Ensure position independent code for CUDA host linking if necessary
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
