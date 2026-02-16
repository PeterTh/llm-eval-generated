cmake_minimum_required(VERSION 3.18)
# Predefine CUDA architectures to avoid detection issues on systems without native nvcc defaults
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile cahn_hilliard.cpp with NVCC as CUDA source so device kernels are compiled
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Compiler options
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link libraries: math, MPI, OpenMP
target_link_libraries(cahn_hilliard PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Ensure OpenMP flags are used when compiling
if(OpenMP_CXX_FOUND)
    target_compile_options(cahn_hilliard PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Enable separable compilation for CUDA
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
