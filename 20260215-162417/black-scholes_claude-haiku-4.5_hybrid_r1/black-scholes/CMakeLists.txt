cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA (optional) - try to find it but don't fail if not found
find_package(CUDA QUIET)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)

# Link libraries
target_link_libraries(black_scholes 
    PRIVATE m 
    PRIVATE MPI::MPI_CXX 
    PRIVATE OpenMP::OpenMP_CXX)

if(CUDA_FOUND)
    target_compile_definitions(black_scholes PRIVATE ENABLE_CUDA=1)
    target_link_libraries(black_scholes PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(black_scholes PRIVATE ${CUDA_INCLUDE_DIRS})
endif()
