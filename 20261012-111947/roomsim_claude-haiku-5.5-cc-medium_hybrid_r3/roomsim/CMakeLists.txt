cmake_minimum_required(VERSION 3.24)

# Must be set before project(): CMake fills the cache with the compiler default otherwise
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Hybrid parallelization: MPI (distributed rows), OpenMP (host threads), CUDA (GPU kernels)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# roomsim.cpp contains device code, so it is compiled as CUDA source
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings. Floating-point contraction is disabled on both the host
# and the device so that every operation rounds the same way on CPU and GPU.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -allow-unsupported-compiler>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra,-Wpedantic,-ffp-contract=off,${OpenMP_CXX_FLAGS}>
)
target_link_options(roomsim PRIVATE ${OpenMP_CXX_FLAGS})

# Link MPI, OpenMP runtime, CUDA runtime, and math library
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX CUDA::cudart m)
