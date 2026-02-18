cmake_minimum_required(VERSION 3.10)

# Build unconditionally with CUDA/NVCC for GPU parallelism
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false --prec-div=true --prec-sqrt=true>
)

# Link math library + CUDA runtime
target_link_libraries(roomsim PRIVATE m CUDA::cudart)
