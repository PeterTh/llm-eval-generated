cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA toolchain; if available enable CUDA paths, otherwise compile CPU-only CUDA-stubbed code
find_package(CUDAToolkit)
find_program(CUDA_COMPILER nvcc)
if (CUDAToolkit_FOUND AND CUDA_COMPILER)
    message(STATUS "Found CUDAToolkit and nvcc: enabling CUDA acceleration")
    # Set a default CUDA architecture to allow configuration on systems without detection
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures")
    endif()
    enable_language(CUDA)
    set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
    set(CUDA_LIBS CUDAToolkit::cudart)
    target_compile_definitions(roomsim PRIVATE USE_CUDA=1)
else()
    message(STATUS "CUDA toolchain not detected (nvcc missing or toolkit incomplete): building without CUDA acceleration")
    set(CUDA_LIBS)
    target_compile_definitions(roomsim PRIVATE USE_CUDA=0)
endif()

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link libraries for hybrid execution
target_link_libraries(roomsim PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_LIBS})

# Define macros to enable MPI/OpenMP code paths
target_compile_definitions(roomsim PRIVATE USE_MPI=1 USE_OPENMP=1)
