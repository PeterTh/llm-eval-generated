cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)
# Ensure NVCC compiles the source as CUDA so kernels are handled
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Compiler options
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-O3 -march=native -Wall -Wextra>)

# Link libraries: MPI, OpenMP, math
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# CUDA flags
if(CMAKE_CUDA_COMPILER)
    set_target_properties(floydwarshall PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "All"
    )
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fopenmp>)
endif()
