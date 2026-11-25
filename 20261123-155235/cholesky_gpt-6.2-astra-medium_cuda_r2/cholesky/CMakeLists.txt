cmake_minimum_required(VERSION 3.17)
project(cholesky_benchmark LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)
target_compile_features(cholesky PRIVATE cxx_std_20)
target_compile_options(cholesky PRIVATE -O3 -Wall -Wextra)
target_link_libraries(cholesky PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
