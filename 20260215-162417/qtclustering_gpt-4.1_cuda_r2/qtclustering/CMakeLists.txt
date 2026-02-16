cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(qtclustering PRIVATE -O3 -Wall -Wextra)
target_link_libraries(qtclustering PRIVATE m)
