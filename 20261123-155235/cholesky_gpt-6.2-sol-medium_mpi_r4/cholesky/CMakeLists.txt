cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "Skip deprecated MPI C++ bindings")
find_package(MPI 3.0 REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX m)
