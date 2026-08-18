cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

# Link math library
target_link_libraries(spmv PRIVATE m)
