cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the .cpp as CUDA so __global__ kernels are supported
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Optimizations: nvcc flags + host compiler flags via -Xcompiler
target_compile_options(floydwarshall PRIVATE
    -O3
    --use_fast_math
    -Xcompiler=-fopenmp
    -Xcompiler=-O3
    -Xcompiler=-march=native
)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(floydwarshall PRIVATE ${MPI_CXX_LIBRARIES} -lgomp m)
