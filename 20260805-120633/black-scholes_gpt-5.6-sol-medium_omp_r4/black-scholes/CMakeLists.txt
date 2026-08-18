cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    -O3
    -march=native
    -fno-math-errno
    -Wall
    -Wextra
)

# OpenMP is required: this benchmark always uses the multicore implementation.
target_link_libraries(black_scholes PRIVATE OpenMP::OpenMP_CXX m)
