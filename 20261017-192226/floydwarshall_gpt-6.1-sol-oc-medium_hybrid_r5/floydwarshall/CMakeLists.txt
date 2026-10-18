cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)

# Pass the OpenMP host flags through nvcc as well as linking its runtime.
string(REPLACE " " "," OPENMP_CUDA_FLAGS "${OpenMP_CXX_FLAGS}")
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=${OPENMP_CUDA_FLAGS},-Wall,-Wextra>)
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
    CUDA::cudart m)
