cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile .cpp as CUDA so nvcc handles __global__ kernels
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

set_property(TARGET qtclustering PROPERTY CUDA_ARCHITECTURES 70 80 86 90)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp,-O3,-march=native>
)

target_include_directories(qtclustering SYSTEM PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(qtclustering ${MPI_CXX_LIBRARIES} m gomp)
