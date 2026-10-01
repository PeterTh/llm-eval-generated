cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Allow vector square roots without changing floating-point arithmetic order.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

# Link math library
target_link_libraries(nbody PRIVATE m OpenMP::OpenMP_CXX)
