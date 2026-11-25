cmake_minimum_required(VERSION 3.20)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
# Forward the OpenMP compile flags to nvcc's host compiler.
target_compile_options(black_scholes PRIVATE -O3
    -Xcompiler=${OpenMP_CXX_FLAGS} -Xcompiler=-march=native)
