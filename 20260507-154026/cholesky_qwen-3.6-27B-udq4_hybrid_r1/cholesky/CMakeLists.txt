cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations; pass OpenMP and native-arch flags to the host compiler via nvcc
target_compile_options(cholesky PRIVATE -O3 -Wall -Wextra
    -Xcompiler=-fopenmp -Xcompiler=-march=native)

# Link MPI, OpenMP runtime, cuSOLVER, cuBLAS, and math library
target_link_libraries(cholesky MPI::MPI_CXX gomp m cublas cusolver)
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_PATH})
