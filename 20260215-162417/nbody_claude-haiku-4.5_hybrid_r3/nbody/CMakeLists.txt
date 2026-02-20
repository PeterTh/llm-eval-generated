cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find CUDA (optional - but will be attempted by language)
if(CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif()

# Create executable
add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -ffast-math)

# Add MPI support
target_link_libraries(nbody PRIVATE MPI::MPI_CXX)
target_compile_definitions(nbody PRIVATE USE_MPI=1)

# Add OpenMP support
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)

# Add CUDA support if available
if(CUDA_FOUND)
    target_compile_definitions(nbody PRIVATE CUDA_ENABLED=1)
endif()

# Link math library
target_link_libraries(nbody PRIVATE m)


