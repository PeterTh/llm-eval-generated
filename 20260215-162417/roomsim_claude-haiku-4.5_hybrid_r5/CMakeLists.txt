cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(roomsim roomsim/roomsim.cpp)

# Include directory for common headers
target_include_directories(roomsim PUBLIC ${CMAKE_SOURCE_DIR})

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link all libraries at once
target_link_libraries(roomsim 
    PUBLIC MPI::MPI_CXX
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC ${CUDA_LIBRARIES}
    PUBLIC m)

# CUDA include directories
target_include_directories(roomsim PUBLIC ${CUDA_INCLUDE_DIRS})

# CUDA compilation flags for maximum performance
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -arch=sm_80 --use_fast_math)

