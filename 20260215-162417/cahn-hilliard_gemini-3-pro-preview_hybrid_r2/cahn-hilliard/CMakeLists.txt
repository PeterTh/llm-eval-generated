cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_70>
)

target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
