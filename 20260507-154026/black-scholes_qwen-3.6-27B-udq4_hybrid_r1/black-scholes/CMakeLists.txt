cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Detect MPI
find_package(MPI REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Force nvcc to compile the .cpp file (contains CUDA kernels)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Compile flags: O3, native, OpenMP, warnings
target_compile_options(black_scholes PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra,-fopenmp)

# Link: MPI, OpenMP, CUDA runtime, math
target_link_libraries(black_scholes
    PRIVATE
        MPI::MPI_CXX
        gomp
        m
)

# Include directories
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_PATH})
