cmake_minimum_required(VERSION 3.10)

project(qtclustering C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Force nvcc for the .cpp source (it contains __global__ CUDA kernels)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# nvcc compile flags
set_target_properties(qtclustering PROPERTIES
    CUDA_ARCHITECTURES OFF
)

target_compile_options(qtclustering PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Xcompiler=-fopenmp
)

# Link: use -Xlinker to pass MPI/OpenMP libs through nvcc
target_link_libraries(qtclustering
    ${MPI_CXX_LIBRARIES}
    cudart
    m
    -fopenmp
)

# Include directories
target_include_directories(qtclustering PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
)


