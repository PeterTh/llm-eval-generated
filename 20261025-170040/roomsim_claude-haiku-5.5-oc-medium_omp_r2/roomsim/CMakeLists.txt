cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required: the benchmark is always built with shared-memory parallelism
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link math library and OpenMP runtime
target_link_libraries(roomsim PRIVATE m OpenMP::OpenMP_CXX)
