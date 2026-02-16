cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
if(OpenMP_CXX_FOUND)
  target_link_libraries(cholesky PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(cholesky PUBLIC MPI::MPI_CXX m cuda)
