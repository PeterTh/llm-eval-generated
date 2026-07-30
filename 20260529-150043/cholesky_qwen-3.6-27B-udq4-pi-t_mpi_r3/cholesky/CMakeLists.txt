cmake_minimum_required(VERSION 3.10)

# Use MPI C++ compiler wrapper to get correct include paths and flags
set(CMAKE_CXX_COMPILER mpicxx)

project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Enable optimizations (mpicxx already links MPI)
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(cholesky m)
