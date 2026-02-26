cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations and CUDA architecture targeting
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native>
)

# Link libraries
target_link_libraries(floydwarshall 
    PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    m
)

# Set CUDA properties
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
