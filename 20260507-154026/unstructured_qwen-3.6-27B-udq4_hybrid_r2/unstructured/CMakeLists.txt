cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Enable CUDA (nvcc will compile the .cpp file)
enable_language(CUDA)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp,-O3,-march=native")

# Main executable – compiled with nvcc for CUDA kernel support
add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES 86)
target_compile_options(unstructured PRIVATE -O3)
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured ${MPI_CXX_LIBRARIES} m -lgomp)
