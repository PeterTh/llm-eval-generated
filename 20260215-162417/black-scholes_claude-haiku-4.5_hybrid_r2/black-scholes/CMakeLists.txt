cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Create executable
add_executable(black_scholes black_scholes.cpp)

# Aggressive optimization flags for maximum performance
target_compile_options(black_scholes PRIVATE 
    -O3 -march=native -mtune=native
    -ffast-math -fno-signed-zeros -fno-trapping-math -frename-registers
    -Wall -Wextra -Wno-unused-parameter
)

# Enable LTO for better optimization across compilation units
set_property(TARGET black_scholes PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)

# Link MPI
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX)

# Link OpenMP with aggressive optimization
target_link_libraries(black_scholes PRIVATE OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(black_scholes PRIVATE m)

# Try to enable OpenMP offloading if available (for potential GPU acceleration)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found - hybrid MPI+OpenMP parallelization enabled")
endif()

message(STATUS "Building with aggressive performance optimizations")






