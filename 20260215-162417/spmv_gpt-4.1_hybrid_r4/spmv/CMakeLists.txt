cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library, MPI, OpenMP, CUDA
if(OpenMP_CXX_FOUND)
  target_link_libraries(spmv PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(spmv PUBLIC MPI::MPI_CXX)
target_link_libraries(spmv PUBLIC CUDA::cudart)
target_link_libraries(spmv m)
