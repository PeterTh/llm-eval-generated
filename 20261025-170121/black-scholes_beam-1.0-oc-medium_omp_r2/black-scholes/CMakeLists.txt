cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP and math library
target_link_libraries(black_scholes PRIVATE OpenMP::OpenMP_CXX m)
