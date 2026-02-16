cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
if(CUDA_FOUND)
    target_link_libraries(cholesky m MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_CUDART_LIBRARY})
    target_compile_definitions(cholesky PRIVATE USE_MPI USE_OPENMP USE_CUDA)
else()
    target_link_libraries(cholesky m MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_compile_definitions(cholesky PRIVATE USE_MPI USE_OPENMP)
endif()
