cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cu)

target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cholesky PRIVATE CUDA::cusolver m)
