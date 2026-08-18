cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# The benchmark always uses CUDA.  The target architecture matches the
# available Ampere GPUs and avoids carrying unnecessary device code variants.
set_target_properties(cholesky PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations.  Keep host warnings separate, since this source is
# compiled by nvcc.
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cholesky PRIVATE CUDA::cudart CUDA::cublas m)
