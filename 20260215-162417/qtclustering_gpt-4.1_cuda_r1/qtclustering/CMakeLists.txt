cmake_minimum_required(VERSION 3.10)

project(qtclustering)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cuda_add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(qtclustering m cuda)
