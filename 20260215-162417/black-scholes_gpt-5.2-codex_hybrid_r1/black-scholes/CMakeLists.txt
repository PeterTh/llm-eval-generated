cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-fopenmp -Xcompiler=-Wall -Xcompiler=-Wextra>
)

# Link math library
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
