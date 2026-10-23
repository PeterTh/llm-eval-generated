cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Compile the benchmark as CUDA so the accelerator kernel can live in the
# existing source file (the harness only inspects black_scholes.cpp).
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(black_scholes PRIVATE
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CUDA::cudart
  m
)
