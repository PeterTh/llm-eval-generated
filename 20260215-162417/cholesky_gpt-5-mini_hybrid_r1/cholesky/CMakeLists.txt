cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Compile as CUDA source so nvcc compiles kernels embedded in the file
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations and OpenMP
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cholesky PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link MPI and CUDA runtimes and math library
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX CUDAToolkit::cudart m)

# Ensure include directories for MPI
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_PATH})
