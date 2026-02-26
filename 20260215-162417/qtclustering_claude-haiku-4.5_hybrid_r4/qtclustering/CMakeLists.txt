cmake_minimum_required(VERSION 3.20)

project(qtclustering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Check if CUDA is available, use it if found
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    set(CUDA_AVAILABLE TRUE)
else()
    set(CUDA_AVAILABLE FALSE)
endif()

add_executable(qtclustering qtclustering.cpp)

# Compilation flags
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA-specific compilation (if available)
if(CUDA_AVAILABLE)
    # Compile with CUDA but let the compiler handle the hybrid code
    set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(qtclustering PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp --expt-relaxed-constexpr>
    )
    target_compile_definitions(qtclustering PRIVATE ENABLE_CUDA)
    target_link_libraries(qtclustering PRIVATE ${CUDA_LIBRARIES})
endif()

# Link required libraries
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
