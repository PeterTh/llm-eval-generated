cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# This benchmark always uses shared-memory OpenMP parallelism.
find_package(OpenMP REQUIRED)

# Link math library and the OpenMP runtime.
target_link_libraries(roomsim PRIVATE m OpenMP::OpenMP_CXX)
