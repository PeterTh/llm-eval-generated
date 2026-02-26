cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Enable CUDA
enable_language(CUDA)

add_executable(qtclustering qtclustering.cu)

set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(qtclustering PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link libraries
target_link_libraries(qtclustering 
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
