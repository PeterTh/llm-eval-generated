cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile the .cpp file with nvcc (it contains CUDA device code)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
# Pass -fopenmp to the host compiler via nvcc's -Xcompiler flag
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# CUDA compile options
set_target_properties(roomsim PROPERTIES CUDA_ARCHITECTURES "70;80;86")
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Link MPI, CUDA runtime, OpenMP, and math libraries
target_link_libraries(roomsim MPI::MPI_CXX cudadevrt gomp m)
