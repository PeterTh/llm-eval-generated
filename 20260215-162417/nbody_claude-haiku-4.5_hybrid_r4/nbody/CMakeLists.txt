cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable MPI
find_package(MPI REQUIRED)

# Enable OpenMP
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cu)

# Enable optimizations and parallelization flags
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -arch=sm_60>)

# Link MPI, OpenMP, and math library
target_link_libraries(nbody PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Define compilation flags for parallelization
target_compile_definitions(nbody PRIVATE USE_MPI USE_CUDA)
