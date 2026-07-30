cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Set CUDA architecture (auto-detect or use common ones)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Main executable - compiled with nvcc since it contains CUDA kernels
add_executable(roomsim roomsim.cpp)

# Set source file language to CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Compile options
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -fopenmp,-march=native,-Wall,-Wextra,-Wpedantic>
)

# Include directories
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(roomsim
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
