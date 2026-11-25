cmake_minimum_required(VERSION 3.20)
project(cholesky_benchmark LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Shared)
target_compile_options(cholesky PRIVATE -O3 -Xcompiler=-Wall,-Wextra)
target_link_libraries(cholesky PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
