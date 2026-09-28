cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations. -ffp-contract=off keeps the compiler from fusing
# multiply-adds on its own; the kernels place the fused operations explicitly so
# that the results stay bit-identical to the sequential reference.
target_compile_options(cholesky PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library and MPI
target_link_libraries(cholesky m MPI::MPI_CXX)
