cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

# Set properties for CUDA compilation
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

# Create executable with CUDA language
add_executable(floydwarshall floydwarshall.cpp)

# Set the source file to be compiled as CUDA
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native>
)

# Include CUDA runtime headers
target_include_directories(floydwarshall PRIVATE ${CUDA_INCLUDE_DIRS})

# Link libraries
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m ${CUDA_LIBRARIES})
