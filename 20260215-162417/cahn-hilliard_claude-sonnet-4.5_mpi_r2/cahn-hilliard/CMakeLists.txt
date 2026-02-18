cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Include MPI headers and link MPI libraries
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(cahn_hilliard m ${MPI_CXX_LIBRARIES})
target_compile_options(cahn_hilliard PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_options(cahn_hilliard PRIVATE ${MPI_CXX_LINK_FLAGS})
