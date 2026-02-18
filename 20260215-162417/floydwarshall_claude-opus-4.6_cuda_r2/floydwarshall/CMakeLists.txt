cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
target_link_libraries(floydwarshall m)
