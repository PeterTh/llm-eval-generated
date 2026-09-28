cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the single source file as CUDA so it can contain both host
# (MPI/OpenMP) and device (CUDA kernel) code.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations. OpenMP and warning flags are forwarded to the
# host compiler via -Xcompiler since this is compiled with nvcc.
target_compile_options(spmv PRIVATE
    -O3
    --use_fast_math
    -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp
)

target_link_libraries(spmv PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
