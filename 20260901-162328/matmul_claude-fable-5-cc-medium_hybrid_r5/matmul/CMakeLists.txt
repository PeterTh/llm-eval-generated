cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The source is a hybrid MPI+OpenMP+CUDA translation unit; compile it with nvcc
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "80;86")

# Enable optimizations (host-side flags forwarded through nvcc)
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>
)

target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP runtime, and math library
target_link_libraries(matmul ${MPI_CXX_LIBRARIES} gomp m)
