cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Detect nvcc (prefer explicit NVCC presence to decide whether to enable CUDA)
find_program(NVCC nvcc)
if(NVCC)
    message(STATUS "nvcc found: ${NVCC}; enabling CUDA build")
    # set a default CUDA architecture if not provided
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES Auto CACHE STRING "CUDA architectures" FORCE)
    endif()
    enable_language(CUDA)
    set(HAVE_CUDA TRUE)
else()
    message(STATUS "nvcc not found; building CPU-only fallback (OpenMP+MPI)")
    set(HAVE_CUDA FALSE)
endif()

add_executable(spmv spmv.cpp)

# If CUDA detected, compile with CUDA support (compile spmv.cpp with NVCC)
if(HAVE_CUDA)
    set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(spmv PRIVATE USE_CUDA=1)
    set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # try to link cudart if available
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND TARGET CUDA::cudart)
        target_link_libraries(spmv PRIVATE CUDA::cudart)
    endif()
else()
    target_compile_definitions(spmv PRIVATE USE_CUDA=0)
endif()

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spmv PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(spmv PRIVATE -fopenmp)
endif()

# Link math and MPI
target_link_libraries(spmv PRIVATE m MPI::MPI_CXX)
