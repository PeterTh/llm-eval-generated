cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA is mandatory. cuBLAS and cuSOLVER are supplied by the CUDA toolkit;
# no additional third-party libraries or CPU implementation are used.
find_package(CUDAToolkit REQUIRED)
add_executable(cholesky cholesky.cpp)
target_compile_options(cholesky PRIVATE -O3 -Wall -Wextra)
target_link_libraries(cholesky PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
