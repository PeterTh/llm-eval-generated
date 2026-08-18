cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra)
elseif(MSVC)
    target_compile_options(black_scholes PRIVATE /O2 /W4)
endif()

# MPI is mandatory: this executable always uses distributed-memory parallelism.
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)
