cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create executable with CUDA support
add_executable(cholesky cholesky.cpp)

# Rename .cpp to .cu to enable CUDA compilation
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA properties
set_target_properties(cholesky PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(cholesky PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")
endif()

# Enable optimizations
target_compile_options(cholesky PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp,-O3,-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Include directories
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(cholesky 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)


