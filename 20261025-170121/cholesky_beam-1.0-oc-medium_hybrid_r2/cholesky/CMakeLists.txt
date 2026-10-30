cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(cholesky PRIVATE ${OpenMP_CXX_FLAGS})

target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cholesky MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)