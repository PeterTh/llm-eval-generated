cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Enable optimizations
# Compile and link with OpenMP for multicore CPU parallelism
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
target_link_options(nbody PRIVATE -fopenmp)

# Link math library
target_link_libraries(nbody m)
