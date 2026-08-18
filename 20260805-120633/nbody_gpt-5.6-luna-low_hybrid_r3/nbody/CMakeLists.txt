cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
