cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# OpenMP shared-memory parallelism
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and OpenMP runtime
target_link_libraries(cholesky m OpenMP::OpenMP_CXX)
