cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations with OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(black_scholes m OpenMP::OpenMP_CXX)
