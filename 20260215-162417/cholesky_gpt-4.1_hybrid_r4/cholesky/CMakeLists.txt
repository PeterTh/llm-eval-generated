cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA libraries
if (CUDA_FOUND)
    target_link_libraries(cholesky m MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
    target_include_directories(cholesky PRIVATE ${CUDA_INCLUDE_DIRS})
else()
    target_link_libraries(cholesky m MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()
