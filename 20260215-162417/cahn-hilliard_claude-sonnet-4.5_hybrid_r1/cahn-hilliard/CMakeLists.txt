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
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Rename source to .cu for CUDA compilation
    set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
    
    add_executable(cahn_hilliard cahn_hilliard.cpp)
    
    # Set CUDA architectures
    set_target_properties(cahn_hilliard PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
    
    # Enable optimizations
    target_compile_options(cahn_hilliard PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
    )
    
    target_compile_definitions(cahn_hilliard PRIVATE USE_CUDA)
    
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(cahn_hilliard 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        CUDA::cudart
        m
    )
else()
    # Build without CUDA
    add_executable(cahn_hilliard cahn_hilliard.cpp)
    
    target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
    
    target_link_libraries(cahn_hilliard 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        m
    )
endif()

# Include MPI headers
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})
