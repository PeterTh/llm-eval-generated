cmake_minimum_required(VERSION 3.24)

# Target the GPUs present on the build/run node. Users may override this
# (e.g. -DCMAKE_CUDA_ARCHITECTURES=80) for a different accelerator cluster.
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(spmv spmv.cpp)

# The benchmark is a single translation unit that contains CUDA kernels as
# well as MPI/OpenMP host code, so compile it with nvcc.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-fopenmp>)

# Enable OpenMP for host code compiled by nvcc and link the runtime.
target_link_options(spmv PRIVATE -fopenmp)

target_link_libraries(spmv PRIVATE
    MPI::MPI_CXX
    ${OpenMP_CXX_LIBRARIES}
    CUDA::cudart
    m)
