cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# MPI is mandatory: there is no serial fallback build or execution path.
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)

# Hardware-native code generation and errno-free square roots improve the hot
# all-pairs kernel without relaxing IEEE floating-point arithmetic.
target_compile_options(nbody PRIVATE
    -O3
    -march=native
    -fno-math-errno
    -Wall
    -Wextra
)
