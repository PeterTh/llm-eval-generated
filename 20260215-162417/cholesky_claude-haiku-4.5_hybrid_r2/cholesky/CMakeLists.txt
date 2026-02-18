cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI, OpenMP, and math library
target_link_libraries(cholesky MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Include MPI headers
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

