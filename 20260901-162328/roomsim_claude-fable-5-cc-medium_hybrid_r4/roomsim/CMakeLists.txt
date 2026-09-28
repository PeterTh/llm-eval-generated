cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 12.0's nvcc rejects GCC > 12 as host compiler; prefer g++-12 if present
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# The single translation unit contains CUDA kernels: compile it as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations, warnings, and host OpenMP
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link with the host C++ compiler; pull in the CUDA runtime explicitly
set_target_properties(roomsim PROPERTIES LINKER_LANGUAGE CXX)
target_link_options(roomsim PRIVATE -fopenmp)
target_link_libraries(roomsim ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} CUDA::cudart m)
