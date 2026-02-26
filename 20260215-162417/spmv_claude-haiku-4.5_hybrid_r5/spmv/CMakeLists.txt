cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA QUIET)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI
target_link_libraries(spmv PRIVATE MPI::MPI_CXX)

# Link OpenMP
target_link_libraries(spmv PRIVATE OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(spmv PRIVATE m)

# Add CUDA support if available
if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling GPU support")
    target_compile_definitions(spmv PRIVATE CUDA_ENABLED)
    target_link_libraries(spmv PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(spmv PRIVATE ${CUDA_INCLUDE_DIRS})
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()
