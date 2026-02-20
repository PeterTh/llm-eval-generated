cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI libraries
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX)

# Link OpenMP libraries
target_link_libraries(cholesky PRIVATE OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(cholesky PRIVATE m)
