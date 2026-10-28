cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required unconditionally for shared-memory parallelism.
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
# -ffp-contract=off keeps floating-point results independent of FMA contraction,
# so output is identical across the blocked/scalar paths, thread counts, and compilers.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link OpenMP runtime and math library
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
