cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(qtclustering qtclustering.cpp)
set_property(SOURCE qtclustering.cpp PROPERTY LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
target_compile_options(qtclustering PRIVATE --use_fast_math)

# Link math library
target_link_libraries(qtclustering m cudart)
