cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Treat .cpp file as CUDA source
add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(spmv 
    PRIVATE MPI::MPI_CXX 
    PRIVATE OpenMP::OpenMP_CXX 
    PRIVATE ${CUDA_LIBRARIES}
    PRIVATE m)

# Include CUDA
target_include_directories(spmv PRIVATE ${CUDA_INCLUDE_DIRS})

# Enable CUDA compilation flags
set_property(TARGET spmv PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(spmv PRIVATE USE_CUDA)
