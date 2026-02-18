cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Try to find CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set source as CUDA
    set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
    
    add_executable(cahn_hilliard cahn_hilliard.cpp)
    
    # Set CUDA architectures
    set_target_properties(cahn_hilliard PROPERTIES 
        CUDA_ARCHITECTURES "60;70;75;80;86"
        CUDA_SEPARABLE_COMPILATION ON)
    
    # Compile options with CUDA
    target_compile_options(cahn_hilliard PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
    )
    
    # Link with CUDA runtime
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(cahn_hilliard 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        CUDA::cudart
        m
    )
    
    message(STATUS "Building with CUDA support")
else()
    # Build without CUDA
    add_executable(cahn_hilliard cahn_hilliard.cpp)
    
    # Compile options without CUDA
    target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
    
    # Link without CUDA
    target_link_libraries(cahn_hilliard 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        m
    )
    
    message(STATUS "Building without CUDA support (CPU fallback)")
endif()

# Include MPI headers
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})
