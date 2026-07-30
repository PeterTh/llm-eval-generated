cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # Ampere (RTX 3090)

# Use mpicxx as the CUDA host compiler for proper MPI integration
find_program(MPI_CXX_COMPILER mpicxx)
if(MPI_CXX_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
endif()

# Force .cpp to be compiled with nvcc (contains CUDA kernels)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Include directories
target_include_directories(cholesky PRIVATE
    ${MPI_CXX_INCLUDE_PATH}
    ${CMAKE_SOURCE_DIR}/../common
)

# Compile options
target_compile_options(cholesky PRIVATE
    -O3 -march=native -Wall -Wextra
)

# Link libraries: cuBLAS, MPI, OpenMP, math
target_link_libraries(cholesky PRIVATE
    cublas
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
