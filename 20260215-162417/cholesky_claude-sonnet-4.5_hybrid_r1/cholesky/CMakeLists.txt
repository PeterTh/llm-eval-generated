cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX)

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
    find_package(CUDAToolkit REQUIRED)
    set(HAVE_CUDA TRUE)
    add_compile_definitions(HAVE_CUDA)
else()
    message(STATUS "CUDA not found - building without GPU support")
    set(HAVE_CUDA FALSE)
endif()

add_executable(cholesky cholesky.cpp)

# Set CUDA compilation if available
if(HAVE_CUDA)
    set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
    
    target_compile_options(cholesky PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -march=native,-fopenmp>
    )
    
    target_link_libraries(cholesky CUDA::cudart)
    
    # Set CUDA architectures
    set_target_properties(cholesky PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86"
    )
endif()

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Add MPI and OpenMP flags
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(cholesky PRIVATE ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})

# Link libraries
target_link_libraries(cholesky 
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
