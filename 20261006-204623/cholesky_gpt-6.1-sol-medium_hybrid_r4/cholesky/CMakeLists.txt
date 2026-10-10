cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source and executable names; nvcc compiles the CUDA kernels.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp,-Wall,-Wextra>)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
    CUDA::cudart CUDA::cublas CUDA::cusolver)
