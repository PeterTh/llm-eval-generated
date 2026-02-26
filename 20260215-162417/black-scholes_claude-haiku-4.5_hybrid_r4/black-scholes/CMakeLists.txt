cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Enable CUDA language support
enable_language(CUDA)

add_executable(black_scholes black_scholes.cpp)

# Set the source file to be compiled with CUDA
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_50 -maxrregcount=32 --extended-lambda>)

# Set CUDA properties
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(black_scholes PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")

# Link libraries
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX)
target_link_libraries(black_scholes PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(black_scholes PRIVATE m)
