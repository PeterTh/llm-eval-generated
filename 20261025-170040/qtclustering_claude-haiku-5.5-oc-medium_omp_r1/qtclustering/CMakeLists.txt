cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required: the program unconditionally uses shared-memory parallelism
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP runtime and math library
target_link_libraries(qtclustering PRIVATE OpenMP::OpenMP_CXX m)
