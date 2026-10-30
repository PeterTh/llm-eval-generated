cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp -Xcompiler=-march=native>
)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cudart)
