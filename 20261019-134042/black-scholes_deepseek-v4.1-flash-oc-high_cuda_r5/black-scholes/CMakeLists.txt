cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU(s) present on the build machine for best performance.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(black_scholes black_scholes.cpp)

# The benchmark is implemented with CUDA in the existing source file.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xcompiler=-O3,-Wall,-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)

find_package(CUDAToolkit REQUIRED)

# Link math library and the CUDA runtime
target_link_libraries(black_scholes m CUDA::cudart)
