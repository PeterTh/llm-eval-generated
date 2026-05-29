cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80 86)

# Build as CUDA executable (compiles .cpp with nvcc for CUDA interop)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    --expt-relaxed-constexpr
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-fopenmp
)

# Link MPI, OpenMP, and math library
target_link_libraries(black_scholes
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})
