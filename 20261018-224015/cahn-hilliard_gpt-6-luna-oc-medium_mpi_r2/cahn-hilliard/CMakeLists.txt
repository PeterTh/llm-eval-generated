cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX m)
