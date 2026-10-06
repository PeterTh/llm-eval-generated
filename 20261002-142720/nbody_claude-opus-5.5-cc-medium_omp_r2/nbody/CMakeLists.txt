cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# OpenMP for shared-memory multicore parallelism
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(nbody PRIVATE m)
