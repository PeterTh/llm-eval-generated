cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(HAVE_CUDA ON)
    message(STATUS "CUDA support enabled")
else()
    set(HAVE_CUDA OFF)
    message(STATUS "CUDA not available - using CPU-only version")
endif()

# Create executable
add_executable(unstructured unstructured.cpp)

# Configure based on CUDA availability
if(HAVE_CUDA)
    # Set CUDA properties
    set_target_properties(unstructured PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
    
    # Enable CUDA compilation for the .cpp file
    set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
    
    # CUDA compile options
    target_compile_options(unstructured PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
    )
    
    target_compile_definitions(unstructured PRIVATE USE_CUDA=1)
    target_link_libraries(unstructured CUDA::cudart)
else()
    target_compile_definitions(unstructured PRIVATE USE_CUDA=0)
endif()

# Include directories
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Common compile options
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(unstructured 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
