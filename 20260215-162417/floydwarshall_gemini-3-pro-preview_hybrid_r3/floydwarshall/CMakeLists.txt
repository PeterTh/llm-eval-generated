cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations and CUDA
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native -Xcompiler -Wall,-Wextra,-fopenmp>
)

# Link libraries
target_link_libraries(floydwarshall 
    PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    m
)

# Treat .cpp as CUDA source
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
