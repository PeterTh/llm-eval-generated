cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)

target_link_libraries(cholesky m)
