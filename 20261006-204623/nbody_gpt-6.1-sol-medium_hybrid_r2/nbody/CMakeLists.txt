cmake_minimum_required(VERSION 3.24)
# Build native GPU code by default; clusters with different GPU models can
# specify a portable architecture list with -DCMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
# CUDA 12 requires a supported host compiler (e.g. GCC 12); select it with
# -DCMAKE_CUDA_HOST_COMPILER when the system compiler is newer.
project(nbody_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
