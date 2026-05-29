cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Find dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Mark the .cpp file as CUDA so nvcc compiles it (it contains __global__ kernels)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
set_target_properties(roomsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic -fopenmp)
target_compile_options(roomsim PRIVATE --expt-relaxed-constexpr)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(roomsim MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)
