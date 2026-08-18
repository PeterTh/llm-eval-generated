cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(floydwarshall floydwarshall.cpp)

# This benchmark always uses its distributed-memory MPI implementation.
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX m)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(floydwarshall PRIVATE
        -O3 -march=native -fomit-frame-pointer -Wall -Wextra)
endif()
