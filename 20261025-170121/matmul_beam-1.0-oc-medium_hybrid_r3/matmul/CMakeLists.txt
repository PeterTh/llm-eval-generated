cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(matmul PRIVATE -O3 -Xcompiler -fopenmp -Xcompiler -march=native)
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(matmul ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
