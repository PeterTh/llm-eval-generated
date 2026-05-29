cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Compile with nvcc for CUDA kernel support
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Xcompiler=-fopenmp -Wall -Wextra -Wpedantic)
set_target_properties(roomsim PROPERTIES CUDA_ARCHITECTURES "86")

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(roomsim MPI::MPI_CXX ${CUDA_cudart_LIBRARY} m)
target_link_options(roomsim PRIVATE -fopenmp)
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_PATH})
