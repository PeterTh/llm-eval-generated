cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and CUDA support
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link libraries
target_link_libraries(matmul PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
