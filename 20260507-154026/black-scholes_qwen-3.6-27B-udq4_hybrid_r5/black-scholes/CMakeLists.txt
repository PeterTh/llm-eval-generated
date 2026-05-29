cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Compile with nvcc since the file contains CUDA device code
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Host compiler optimizations
set_target_properties(black_scholes PROPERTIES
    CUDA_HOST_COMPILER g++
)

# Enable optimizations for host and device
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3;-Xcompiler=-march=native,-Wall,-Wextra,-fopenmp>
)

# Link MPI, OpenMP, and math library
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(black_scholes
    ${MPI_CXX_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    m
)
