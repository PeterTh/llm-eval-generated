cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for shared-memory multicore parallelism (compiler-provided, no external dependency)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(matmul matmul.cpp)

# Enable optimizations.  -ffp-contract=off keeps floating-point rounding
# explicit (see multiplyTile in matmul.cpp) so results are reproducible.
target_compile_options(matmul PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library and OpenMP runtime
target_link_libraries(matmul PRIVATE m OpenMP::OpenMP_CXX)
