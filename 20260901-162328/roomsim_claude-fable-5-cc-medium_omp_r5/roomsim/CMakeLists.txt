cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# OpenMP for multicore shared-memory parallelism
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link math library and OpenMP
target_link_libraries(roomsim m OpenMP::OpenMP_CXX)
