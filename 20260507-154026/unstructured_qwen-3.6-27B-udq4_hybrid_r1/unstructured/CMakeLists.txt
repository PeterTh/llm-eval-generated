cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Enable CUDA (nvcc will compile the .cpp file)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Host compile options (passed to g++ via nvcc)
set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(unstructured PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Xcompiler="-fopenmp"
)

# Link MPI, OpenMP, CUDA runtime, math
target_link_libraries(unstructured MPI::MPI_CXX -fopenmp ${CUDA_cudart_LIBRARY} m)

# Include MPI headers
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_PATH})
