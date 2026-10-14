cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required: the parallel code path is always used
find_package(OpenMP REQUIRED)

# Link OpenMP runtime and math library
target_link_libraries(qtclustering OpenMP::OpenMP_CXX m)
