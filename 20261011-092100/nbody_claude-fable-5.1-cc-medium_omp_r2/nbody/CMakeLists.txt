cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required: the program unconditionally uses shared-memory parallelism
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations. FMA contraction is disabled so floating-point
# evaluation is fixed by the source (explicit std::fma where intended) and
# does not depend on how the compiler outlines the OpenMP regions.
target_compile_options(nbody PRIVATE -O3 -march=native -ffp-contract=off -Wall -Wextra)

# Link math library
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
