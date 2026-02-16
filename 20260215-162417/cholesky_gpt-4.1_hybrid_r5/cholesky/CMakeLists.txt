cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(cholesky PRIVATE USE_MPI USE_OPENMP USE_CUDA)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cuda cudart)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# Link math library
target_link_libraries(cholesky m)
