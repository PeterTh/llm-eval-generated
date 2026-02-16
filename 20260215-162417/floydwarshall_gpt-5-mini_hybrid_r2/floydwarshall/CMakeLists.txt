cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(floydwarshall floydwarshall.cpp)

# Building CPU-only fallback (MPI+OpenMP). CUDA kernels are disabled in this build.
message(STATUS "Building with MPI+OpenMP; CUDA kernels disabled (CPU fallback)")

# Enable optimizations for C++
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)

# Link libraries
if(TARGET MPI::MPI_CXX)
    target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
else()
    target_link_libraries(floydwarshall PRIVATE OpenMP::OpenMP_CXX m)
endif()

target_include_directories(floydwarshall PRIVATE ${MPI_INCLUDE_PATH})
