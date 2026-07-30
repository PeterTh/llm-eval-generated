cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI and math libraries
target_link_libraries(nbody MPI::MPI_CXX m)
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_PATH})
