cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
set_property(TARGET cholesky PROPERTY CUDA_ARCHITECTURES native)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cholesky m CUDA::cublas CUDA::cusolver)
