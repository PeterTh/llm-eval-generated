cmake_minimum_required(VERSION 3.20)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(black_scholes PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(black_scholes PRIVATE -O3
    "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=${OpenMP_CXX_FLAGS},-march=native>"
    "$<$<COMPILE_LANG_AND_ID:CUDA,Clang>:${OpenMP_CXX_FLAGS};-march=native>")
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
