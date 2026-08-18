cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required: this benchmark always evaluates seed candidates in
# parallel shared-memory workers.
find_package(OpenMP REQUIRED)
target_link_libraries(qtclustering PRIVATE m OpenMP::OpenMP_CXX)
