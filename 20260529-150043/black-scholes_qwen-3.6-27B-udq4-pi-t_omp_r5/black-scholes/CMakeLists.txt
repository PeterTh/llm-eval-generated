cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations and OpenMP
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(black_scholes PRIVATE OpenMP::OpenMP_CXX m)
