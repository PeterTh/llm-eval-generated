cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(HAVE_CUDA ON)
    add_compile_definitions(HAVE_CUDA)
    message(STATUS "CUDA support enabled")
else()
    set(HAVE_CUDA OFF)
    message(STATUS "CUDA not available - using CPU fallback")
endif()

if(HAVE_CUDA)
    # Rename source file to .cu for CUDA compilation
    set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(floydwarshall floydwarshall.cpp)

if(HAVE_CUDA)
    # Set CUDA architectures (support common GPU architectures)
    set_target_properties(floydwarshall PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    )
endif()

# Include directories
target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Compile options
if(HAVE_CUDA)
    target_compile_options(floydwarshall PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -fopenmp>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
    )
else()
    target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
endif()

# Link libraries
if(HAVE_CUDA)
    target_link_libraries(floydwarshall 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        CUDA::cudart
        m
    )
else()
    target_link_libraries(floydwarshall 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        m
    )
endif()
