cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before project (adjust for your target GPU)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# Set source file to be compiled as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -fopenmp>
)

# Link libraries
target_link_libraries(roomsim 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Include directories
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
