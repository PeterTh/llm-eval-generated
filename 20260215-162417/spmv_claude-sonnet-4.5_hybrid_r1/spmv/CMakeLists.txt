cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark)

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
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    add_definitions(-DUSE_CUDA)
    
    # Set source properties for CUDA
    set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
    
    add_executable(spmv spmv.cpp)
    
    target_compile_options(spmv PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    )
else()
    message(STATUS "CUDA not found, building CPU-only version with MPI+OpenMP")
    
    add_executable(spmv spmv.cpp)
    
    target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link libraries
target_link_libraries(spmv 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    m
)
