cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required unconditionally
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations. -ffp-contract=off keeps floating-point results identical
# regardless of how the compiler outlines the OpenMP regions (no implicit FMA),
# so parallel and serial execution produce bit-identical output.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link OpenMP runtime and math library
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
