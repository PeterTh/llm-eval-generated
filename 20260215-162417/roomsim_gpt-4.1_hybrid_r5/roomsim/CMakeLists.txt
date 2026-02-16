cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

# Link math library, MPI, OpenMP, CUDA
if (OpenMP_FOUND)
    target_link_libraries(roomsim m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
else()
    target_link_libraries(roomsim m MPI::MPI_CXX CUDA::cudart)
endif()
