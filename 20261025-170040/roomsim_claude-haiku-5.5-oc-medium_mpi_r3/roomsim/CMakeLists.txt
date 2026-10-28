cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI is required: the benchmark always runs distributed
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# MPI and math libraries
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX m)
