cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Force nvcc to compile nbody.cpp so CUDA device code is recognized
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(nbody
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

# Include directories
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
