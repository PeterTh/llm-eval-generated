cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)

add_executable(spmv spmv.cu)

# Set CUDA compilation flags to include OpenMP
set_target_properties(spmv PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80"
)

# Enable optimizations
target_compile_options(spmv PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
)

# Include directories
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(spmv 
    m 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
)
