cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    -O3 -march=native -fno-math-errno -fno-trapping-math -Wall -Wextra)

# MPI is mandatory: this benchmark always executes as an MPI program.
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m)

include(CheckIPOSupported)
check_ipo_supported(RESULT BLACK_SCHOLES_IPO_SUPPORTED)
if(BLACK_SCHOLES_IPO_SUPPORTED)
    set_property(TARGET black_scholes PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
