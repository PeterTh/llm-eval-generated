cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile .cpp with nvcc to support CUDA device code
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Optimizations: -O3 for nvcc, pass host compiler flags via -Xcompiler
target_compile_options(nbody PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra,-Wno-unknown-pragmas)

# Link MPI, OpenMP, and math library
target_link_libraries(nbody MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_PATH})
