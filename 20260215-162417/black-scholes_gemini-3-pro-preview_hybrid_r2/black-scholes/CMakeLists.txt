cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# Link libraries
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Set source file properties if needed, though usually automatic with .cu extension.
# Since we can't rename, we might need to tell CMake it's CUDA or rely on mixed mode if supported.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
