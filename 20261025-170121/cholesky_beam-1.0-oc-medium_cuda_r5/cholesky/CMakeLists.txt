cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(cholesky cusolver)
