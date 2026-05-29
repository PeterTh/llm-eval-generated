cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES LINKER_LANGUAGE CXX)

# Enable optimizations; pass -fopenmp through nvcc to host compiler
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp)

# Link MPI, OpenMP runtime, and math library
target_link_libraries(nbody MPI::MPI_CXX ${MPI_CXX_LIBRARIES} gomp m)
