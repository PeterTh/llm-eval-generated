cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)
