cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(matmul PRIVATE cxx_std_20)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-fopenmp> $<$<COMPILE_LANGUAGE:CUDA>:-O3> )
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
