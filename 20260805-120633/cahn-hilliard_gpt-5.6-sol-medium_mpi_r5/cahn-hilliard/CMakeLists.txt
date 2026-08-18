cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(cahn_hilliard cahn_hilliard.cpp)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX m)
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
