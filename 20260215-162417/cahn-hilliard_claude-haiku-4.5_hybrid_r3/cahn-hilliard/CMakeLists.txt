cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI (required)
find_package(MPI REQUIRED)

# Find OpenMP (required)
find_package(OpenMP REQUIRED)

# Try to find CUDA (optional)
find_package(CUDA QUIET)

# Create executable
if(CUDA_FOUND)
    # Enable CUDA support
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    add_executable(cahn_hilliard cahn_hilliard.cpp)
    set_property(TARGET cahn_hilliard PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    # Enable optimizations for CUDA
    target_compile_options(cahn_hilliard PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80>
    )
else()
    # CPU-only build
    add_executable(cahn_hilliard cahn_hilliard.cpp)
    
    target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link libraries
target_link_libraries(cahn_hilliard 
    PRIVATE MPI::MPI_CXX
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE m
)

# Link CUDA if available
if(CUDA_FOUND)
    target_link_libraries(cahn_hilliard PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(cahn_hilliard PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# Print build info
if(CUDA_FOUND)
    message(STATUS "Building with CUDA support")
else()
    message(STATUS "Building without CUDA support")
endif()


