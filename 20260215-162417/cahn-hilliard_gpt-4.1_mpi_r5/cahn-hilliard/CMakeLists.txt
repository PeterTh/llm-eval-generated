cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and MPI libraries
target_link_libraries(cahn_hilliard m MPI::MPI_CXX)

find_package(MPI REQUIRED)
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX)
