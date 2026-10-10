cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
endif()
project(roomsim_benchmark LANGUAGES CXX)
find_package(CUDAToolkit REQUIRED)
# CUDA 12.0-12.3 needs GCC <= 12, even when the system C++ compiler is newer.
# An explicitly selected CUDA host compiler always takes precedence.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13
   AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12
   AND CUDAToolkit_VERSION VERSION_LESS 12.4)
    find_program(ROOMSIM_CUDA_HOST_COMPILER NAMES g++-12 REQUIRED)
    set(CMAKE_CUDA_HOST_COMPILER "${ROOMSIM_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler")
endif()
enable_language(CUDA)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the existing source and executable names, compiling the kernels with nvcc.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
