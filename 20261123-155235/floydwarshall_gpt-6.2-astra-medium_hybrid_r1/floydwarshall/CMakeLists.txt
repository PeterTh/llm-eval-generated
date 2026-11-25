cmake_minimum_required(VERSION 3.20)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(floydwarshall PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(floydwarshall PRIVATE -O3
    "$<$<CUDA_COMPILER_ID:NVIDIA>:-Xcompiler=${OpenMP_CXX_FLAGS}>"
    "$<$<CUDA_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>")
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
