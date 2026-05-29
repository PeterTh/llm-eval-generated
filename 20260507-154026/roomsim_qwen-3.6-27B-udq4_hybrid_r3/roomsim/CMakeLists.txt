cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Set source file language to CUDA so nvcc compiles it
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Set include directories
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS} ../common)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    -O3 -march=native -Wall -Wextra
    -Xcompiler="-fopenmp"
)
target_link_libraries(roomsim
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_LINK_FLAGS}
    -fopenmp
    m
)

# CUDA architecture auto-detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
