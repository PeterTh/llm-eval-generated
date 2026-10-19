cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
set_target_properties(nbody PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Enable optimizations
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(nbody PRIVATE m)
