cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create executable - treat .cpp as CUDA source
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Set CUDA source file property
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math --expt-relaxed-constexpr>
)

# Include directories
target_include_directories(cahn_hilliard PRIVATE 
    ${CMAKE_SOURCE_DIR}/../common
    ${MPI_CXX_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cahn_hilliard 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

# Set CUDA architecture (adjust for your GPU)
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "86"
)
