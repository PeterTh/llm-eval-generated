cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Keep the reference's separate multiply/add operations (including its hash).
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
# OpenMP's imported target supplies the host link flags; nvcc also needs the
# host compile flag when compiling this existing .cpp file as CUDA.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
