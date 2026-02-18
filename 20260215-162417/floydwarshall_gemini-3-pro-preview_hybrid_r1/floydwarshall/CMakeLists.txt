cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES 70)
project(floydwarshall_benchmark LANGUAGES CXX)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Treat .cpp file as CUDA source
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall,-Wextra,-fopenmp>
)

# Link libraries
target_link_libraries(floydwarshall PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    CUDA::cudart
)
