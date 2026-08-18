cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA so the executable remains named
# "cholesky" while containing both host MPI/OpenMP orchestration and kernels.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

set_target_properties(cholesky PROPERTIES
    CUDA_ARCHITECTURES "80-real;80-virtual"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    # The source is compiled by nvcc, so forward OpenMP explicitly to nvcc's
    # host compiler as well as linking OpenMP below.
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--allow-unsupported-compiler;-Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>)

target_link_libraries(cholesky PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    m)
