cmake_minimum_required(VERSION 3.10)

# nvcc isn't on PATH in this environment; point CMake at it explicitly.
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Compile the existing .cpp as CUDA (no new files).
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

find_package(CUDAToolkit REQUIRED)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native>
)

# Prefer building for the local GPU when supported by this CMake version.
set_property(TARGET black_scholes PROPERTY CUDA_ARCHITECTURES native)

# Link math + CUDA runtime
target_link_libraries(black_scholes PRIVATE m CUDA::cudart)
