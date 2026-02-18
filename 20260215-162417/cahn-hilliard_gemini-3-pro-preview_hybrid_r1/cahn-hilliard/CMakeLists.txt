cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# Link libraries
target_link_libraries(cahn_hilliard m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Set CUDA architecture (adjust as needed, e.g., auto or specific arch)
set_property(TARGET cahn_hilliard PROPERTY CUDA_ARCHITECTURES OFF)
