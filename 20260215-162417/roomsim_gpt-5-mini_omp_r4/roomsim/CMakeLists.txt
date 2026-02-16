cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Find and enable OpenMP for parallelization
find_package(OpenMP REQUIRED)

# Enable optimizations and warnings and OpenMP
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic -fopenmp)

# Link OpenMP and math library
target_link_libraries(roomsim PRIVATE OpenMP::OpenMP_CXX m)
