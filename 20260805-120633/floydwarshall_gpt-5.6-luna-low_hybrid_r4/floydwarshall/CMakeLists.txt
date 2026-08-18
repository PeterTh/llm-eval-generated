cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 60)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(floydwarshall PRIVATE m)
