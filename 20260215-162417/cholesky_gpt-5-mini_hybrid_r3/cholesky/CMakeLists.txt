cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Tell CMake to compile the .cpp with the CUDA compiler so kernels in the same file are built
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable aggressive optimizations and OpenMP
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler;-fopenmp;-arch=sm_70>
)

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)

# Ensure CUDA runtime linking
if(NOT TARGET cudart)
    find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)
    if(CUDART_LIB)
        target_link_libraries(cholesky PRIVATE ${CUDART_LIB})
    endif()
endif()
