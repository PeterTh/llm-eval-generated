cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

target_compile_options(floydwarshall PRIVATE --expt-relaxed-constexpr -O3 -Wall -Wextra)

target_link_libraries(floydwarshall m)
