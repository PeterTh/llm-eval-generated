cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Cover current data-centre and workstation GPUs without depending on a GPU
# being visible at configure time.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(nbody nbody.cpp)

# The benchmark lives in a single translation unit, so compile it as CUDA to
# keep the CUDA kernels next to the MPI/OpenMP host orchestration.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
