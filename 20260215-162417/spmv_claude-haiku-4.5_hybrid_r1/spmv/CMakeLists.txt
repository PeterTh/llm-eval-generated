cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Define preprocessor flags for parallelization
set(SPMV_DEFINITIONS USE_MPI USE_OPENMP)

# Try to find CUDA, but make it completely optional
if(COMMAND enable_language)
    # Check if CUDA is available
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        list(APPEND SPMV_DEFINITIONS USE_CUDA)
        set(HAVE_CUDA TRUE)
    endif()
endif()

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Apply definitions
target_compile_definitions(spmv PRIVATE ${SPMV_DEFINITIONS})

# Link libraries
target_link_libraries(spmv PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Link CUDA libraries if found
if(HAVE_CUDA)
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_60>)
    target_link_libraries(spmv PRIVATE cuda cusparse cudart)
endif()
