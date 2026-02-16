cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable (source forced to be compiled with CUDA/NVCC)
add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Compiler options for host and CUDA
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-fopenmp;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link libraries
target_link_libraries(black_scholes PRIVATE MPI::MPI_C OpenMP::OpenMP_CXX CUDAToolkit::cudart m)

# Ensure include directories for MPI
if(MPI_C_FOUND)
    target_include_directories(black_scholes PRIVATE ${MPI_C_INCLUDE_PATH})
endif()
