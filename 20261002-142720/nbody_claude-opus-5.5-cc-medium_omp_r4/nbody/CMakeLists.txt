cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -fno-math-errno -Wall -Wextra)

# Link math library
target_link_libraries(nbody m)

# OpenMP (required)
find_package(OpenMP REQUIRED)
target_link_libraries(nbody OpenMP::OpenMP_CXX)
