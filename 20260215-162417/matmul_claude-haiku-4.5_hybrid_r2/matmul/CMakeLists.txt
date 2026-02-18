cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI (required for distributed computing)
find_package(MPI REQUIRED COMPONENTS CXX)

# Find OpenMP (optional, for intra-node parallelization)
find_package(OpenMP)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI
target_link_libraries(matmul PUBLIC MPI::MPI_CXX)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link math library
target_link_libraries(matmul PUBLIC m)
