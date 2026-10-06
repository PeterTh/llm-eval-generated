cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)

# Single source with host (MPI/OpenMP) and device (CUDA) code: compile with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(spmv PROPERTIES LINKER_LANGUAGE CXX)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(spmv PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
