cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Set CUDA architecture (auto-detect native)
set(CMAKE_CUDA_ARCHITECTURES OFF)

# Find MPI (CXX component)
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Build the executable — compile the .cpp with nvcc so CUDA device code is included
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Include directories
target_include_directories(black_scholes PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/..
)

# Host compiler flags (passed via -Xcompiler to nvcc)
set_target_properties(black_scholes PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Use CUDA_HOST_COMPILER to ensure nvcc uses g++ for host code
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Compile options: nvcc flags + -Xcompiler for host flags
target_compile_options(black_scholes PRIVATE
    -O3
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-fopenmp
)

# Link libraries: MPI, OpenMP, math
target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
