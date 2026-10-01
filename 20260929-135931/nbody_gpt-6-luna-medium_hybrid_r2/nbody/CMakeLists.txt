cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX ${OpenMP_CXX_LIBRARIES} CUDA::cudart)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_definitions(nbody PRIVATE USE_MPI_CUDA_OPENMP=1)

# Enable optimizations
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler=-fopenmp,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(nbody m)
