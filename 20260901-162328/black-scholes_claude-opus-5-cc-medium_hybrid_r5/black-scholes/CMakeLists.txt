cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark source contains CUDA kernels; compile it with nvcc.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations. Host code keeps -O3 -march=native; device code disables FMA
# contraction and uses IEEE-correct division/sqrt to stay numerically faithful to the
# original scalar formulation.
target_compile_options(black_scholes PRIVATE
  -O3
  --fmad=false
  -prec-div=true
  -prec-sqrt=true
  --extended-lambda
  -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra
)

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_link_options(black_scholes PRIVATE -fopenmp)
