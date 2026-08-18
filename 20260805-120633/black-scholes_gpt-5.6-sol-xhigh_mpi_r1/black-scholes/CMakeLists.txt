cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MPI is the benchmark's unconditional distributed-memory execution model.
set(MPI_CXX_SKIP_MPICXX ON CACHE BOOL
    "Use MPI's C interface from C++ without legacy MPI C++ bindings" FORCE)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -fno-math-errno
                                             -Wall -Wextra)

# Link MPI and the platform math library.
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)
