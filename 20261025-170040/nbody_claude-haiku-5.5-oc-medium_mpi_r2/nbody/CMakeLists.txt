cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs in distributed-memory mode
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations. -ffp-contract=off keeps floating-point results independent of
# how the compiler vectorizes/fuses the kernel, so output is identical for any rank count.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library and MPI
target_link_libraries(nbody PRIVATE m MPI::MPI_CXX)
