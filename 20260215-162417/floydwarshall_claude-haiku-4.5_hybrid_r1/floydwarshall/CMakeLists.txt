cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
set(HAVE_CUDA FALSE)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA compiler found and enabled")
endif()

add_executable(floydwarshall floydwarshall.cpp)

# Configure based on CUDA availability
if(HAVE_CUDA)
    message(STATUS "Building with GPU acceleration support")
    set_target_properties(floydwarshall PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # Detect CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 90)
    endif()
    set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
else()
    message(STATUS "Building CPU-only version (CUDA not available)")
endif()

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(floydwarshall MPI::MPI_CXX)
target_link_libraries(floydwarshall OpenMP::OpenMP_CXX)
target_link_libraries(floydwarshall m)



