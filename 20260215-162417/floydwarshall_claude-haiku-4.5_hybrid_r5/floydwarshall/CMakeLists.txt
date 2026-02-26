cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA
find_package(CUDA QUIET)

# Add executable with C++ source
add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Add CUDA sources to target
    target_sources(floydwarshall PRIVATE floydwarshall_cuda.cu)
    
    # Set CUDA compile options
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    
    # Link CUDA libraries
    target_link_libraries(floydwarshall PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(floydwarshall PRIVATE ${CUDA_INCLUDE_DIRS})
    
    # Define CUDA availability
    target_compile_definitions(floydwarshall PRIVATE HAVE_CUDA=1)
else()
    # No CUDA available - compile without it
    target_compile_definitions(floydwarshall PRIVATE HAVE_CUDA=0)
endif()

# Link MPI and OpenMP
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
