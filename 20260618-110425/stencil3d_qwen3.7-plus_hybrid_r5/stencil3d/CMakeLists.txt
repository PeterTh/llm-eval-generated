cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Tell CMake to compile the .cpp file with nvcc (since it contains CUDA kernels)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

add_executable(stencil3d stencil3d.cpp)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Add OpenMP flags
target_compile_options(stencil3d PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(stencil3d PRIVATE ${OpenMP_CXX_FLAGS})

# Add MPI
target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(stencil3d ${MPI_CXX_LIBRARIES})

# Add CUDA
target_include_directories(stencil3d PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(stencil3d CUDA::cudart)

# Set CUDA architecture (adjust as needed for your GPU)
set_target_properties(stencil3d PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Link math library
target_link_libraries(stencil3d m)
