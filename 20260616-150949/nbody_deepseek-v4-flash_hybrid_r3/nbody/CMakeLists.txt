cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find MPI
find_package(MPI REQUIRED)

# Enable CUDA with the host compiler
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Compile nbody.cpp as CUDA source to support __global__ kernels
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link dependencies
target_link_libraries(nbody PRIVATE
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    ${CMAKE_CUDA_LIBRARIES}
)

# Include MPI and CUDA paths
target_include_directories(nbody PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
)
