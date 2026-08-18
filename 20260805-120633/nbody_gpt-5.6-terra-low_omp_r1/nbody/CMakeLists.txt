cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# OpenMP is required: this benchmark always uses shared-memory parallelism.
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
