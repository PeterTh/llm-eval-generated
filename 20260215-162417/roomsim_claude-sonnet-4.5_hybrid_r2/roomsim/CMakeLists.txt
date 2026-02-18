cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check for CUDA (optional)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_ENABLED TRUE)
    set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
    add_definitions(-DUSE_CUDA)
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA not found, building CPU-only version")
endif()

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
if(CUDA_ENABLED)
    target_compile_options(roomsim PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math --expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    )
    
    # Set CUDA architectures
    set_target_properties(roomsim PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86"
    )
    
    # Link CUDA libraries
    target_link_libraries(roomsim 
        m 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        CUDA::cudart
        CUDA::curand
    )
else()
    target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
    
    # Link without CUDA
    target_link_libraries(roomsim 
        m 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
    )
endif()
