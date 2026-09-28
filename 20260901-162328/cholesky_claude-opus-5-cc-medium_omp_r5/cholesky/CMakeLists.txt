cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations. -ffp-contract=off keeps the compiler from fusing the products
# of the dot products in cholesky.cpp: the kernels there reproduce the rounding of the
# original sequential code explicitly (separate products, std::fma where the original
# uses one), which is what makes the parallel results bit-for-bit identical.
target_compile_options(cholesky PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library and OpenMP
target_link_libraries(cholesky m OpenMP::OpenMP_CXX)
