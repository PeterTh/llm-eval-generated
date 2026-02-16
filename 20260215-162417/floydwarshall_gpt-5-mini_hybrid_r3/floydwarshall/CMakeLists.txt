cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Require MPI, OpenMP and CUDA toolkits
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Compile floydwarshall.cpp with CUDA compiler so kernels in the file are supported
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
# Enable OpenMP for host compilation
target_compile_options(floydwarshall PRIVATE ${OpenMP_CXX_FLAGS})

# Link required libraries
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDAToolkit::cudart m)

# Expose defines to code
target_compile_definitions(floydwarshall PRIVATE USE_MPI=1 USE_OPENMP=1 USE_CUDA=1)

# Allow CUDA separable compilation for better optimization
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
