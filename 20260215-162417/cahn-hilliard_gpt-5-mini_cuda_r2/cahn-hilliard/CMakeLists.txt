cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)
# Ensure the .cpp is compiled with NVCC so CUDA kernels are recognized
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for host and device
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
# Additional CUDA-specific flags (best-effort portability)
if(CMAKE_CUDA_COMPILER)
  target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

# Find CUDA runtime
find_package(CUDAToolkit REQUIRED)

# Link CUDA runtime and math library
target_link_libraries(cahn_hilliard PRIVATE CUDA::cudart m)
