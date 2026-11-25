cmake_minimum_required(VERSION 3.20)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(spmv PRIVATE -O3
    $<$<CUDA_COMPILER_ID:NVIDIA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<CUDA_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>)
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
