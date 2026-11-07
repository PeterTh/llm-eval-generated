cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required: the program unconditionally uses OpenMP for
# shared-memory multicore parallelism.
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP and math library
target_link_libraries(cholesky PRIVATE OpenMP::OpenMP_CXX m)
