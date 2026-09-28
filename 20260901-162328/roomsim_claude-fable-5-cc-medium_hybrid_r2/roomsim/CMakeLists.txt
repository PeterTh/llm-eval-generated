cmake_minimum_required(VERSION 3.18)

# nvcc needs a compatible host compiler; prefer g++-12 when the default gcc is newer
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

# The single translation unit contains CUDA kernels; compile it with nvcc
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings (host flags forwarded through nvcc)
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
)

# MPI includes passed directly so nvcc does not receive host-only flags like -pthread
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link with the host C++ compiler (no separable device compilation is used)
set_target_properties(roomsim PROPERTIES LINKER_LANGUAGE CXX)

# Link math library, MPI, OpenMP, and the CUDA runtime
target_link_libraries(roomsim PRIVATE
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
