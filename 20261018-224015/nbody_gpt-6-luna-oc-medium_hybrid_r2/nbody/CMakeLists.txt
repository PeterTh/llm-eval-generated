cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp,-march=native,-Wall,-Wextra>)
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;90")
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(nbody PRIVATE m)
