cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP is required: the solver always executes its stencil sweeps through
# OpenMP worksharing constructs.
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and the required shared-memory parallel runtime.
target_link_libraries(cahn_hilliard PRIVATE m OpenMP::OpenMP_CXX)
