cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source name while compiling its GPU kernels with nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
# OpenMP's imported target supplies CXX flags, so forward them to nvcc's host compiler.
separate_arguments(NBODY_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS NBODY_OPENMP_FLAGS)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

# Link math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
