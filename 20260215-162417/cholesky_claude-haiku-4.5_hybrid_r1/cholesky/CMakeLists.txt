cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find OpenMP and MPI (required)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Try to find CUDA (optional) - don't call find_package which might cause issues
find_program(CUDA_COMPILER nvcc QUIET)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(cholesky 
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE MPI::MPI_CXX
    PRIVATE m
)

# CUDA support (if available)
if(CUDA_COMPILER)
    target_compile_definitions(cholesky PRIVATE ENABLE_CUDA)
    # Try to find CUDA libraries
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        target_link_libraries(cholesky PRIVATE ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
    endif()
endif()
