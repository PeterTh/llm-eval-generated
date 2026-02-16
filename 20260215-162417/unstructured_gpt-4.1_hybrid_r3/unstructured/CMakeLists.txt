cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Enable OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(unstructured PUBLIC OpenMP::OpenMP_CXX)
endif()

# Enable CUDA
enable_language(CUDA)
target_sources(unstructured PRIVATE unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link MPI
target_link_libraries(unstructured PUBLIC MPI::MPI_CXX)

# Link math library
target_link_libraries(unstructured m)
