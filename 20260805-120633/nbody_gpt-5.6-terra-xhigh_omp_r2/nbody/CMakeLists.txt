cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP is required: the simulation's independent body updates execute in
# parallel on shared-memory CPU cores.
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
