cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Permit SIMD square roots while retaining normal floating-point semantics.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

# OpenMP is required for the simulation; there is no serial build mode.
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
