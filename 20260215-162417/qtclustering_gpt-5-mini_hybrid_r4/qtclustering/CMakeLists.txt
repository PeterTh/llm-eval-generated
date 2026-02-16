cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

# Optimization and warnings
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA is optional. To enable CUDA compilation, configure with -DUSE_CUDA=ON and ensure CUDACXX is set in the environment.
if (USE_CUDA)
    message(STATUS "Attempting to enable CUDA compilation (USE_CUDA=ON)")
    # Try to enable CUDA language; if this fails, user should configure without USE_CUDA
    enable_language(CUDA)
    set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(qtclustering PRIVATE USE_CUDA)
    # Link against CUDA runtime and math
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
    set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "CUDA disabled (USE_CUDA not set); compiling with CPU-only GPU fallback")
    target_compile_definitions(qtclustering PRIVATE NO_CUDA)
    target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
endif()
