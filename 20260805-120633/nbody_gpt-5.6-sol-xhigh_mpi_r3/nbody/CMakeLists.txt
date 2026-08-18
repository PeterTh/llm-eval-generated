cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# MPI is unconditional; SIMD-only OpenMP pragmas do not add a runtime or any
# dependency, and fno-math-errno permits vector square roots for positive
# softened distances.
target_compile_options(nbody PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nbody PRIVATE -fno-math-errno -fopenmp-simd)
endif()

target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)
