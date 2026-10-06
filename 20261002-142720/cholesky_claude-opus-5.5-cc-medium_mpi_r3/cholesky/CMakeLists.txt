cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# Enable optimizations (fp-contract off: rounding of the reference build is reproduced explicitly)
target_compile_options(cholesky PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(cholesky MPI::MPI_CXX m)
