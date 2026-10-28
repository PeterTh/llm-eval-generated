cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# OpenMP is required (shared-memory parallelization is always enabled)
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and OpenMP runtime
target_link_libraries(cahn_hilliard PRIVATE m OpenMP::OpenMP_CXX)
