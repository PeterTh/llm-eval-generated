cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# OpenMP is required; the parallel code path is always built
find_package(OpenMP REQUIRED)

# Enable optimizations. -ffp-contract=off keeps floating-point results identical
# across thread counts by preventing FMA fusion that varies with code layout.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library and OpenMP runtime
target_link_libraries(nbody PRIVATE m OpenMP::OpenMP_CXX)
