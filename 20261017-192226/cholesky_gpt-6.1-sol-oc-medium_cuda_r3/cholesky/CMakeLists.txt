cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

# Keep full double precision; fast-math would weaken the validation semantics.
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)

# These libraries are supplied by the required CUDA toolkit.
target_link_libraries(cholesky PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
