cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Device code targets: Volta through Hopper. Override with -DCMAKE_CUDA_ARCHITECTURES=<list>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

# Hybrid parallelization is mandatory: MPI, OpenMP and CUDA are all required
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# roomsim.cpp contains both host and device code, so it is compiled by nvcc.
# Host-side optimizations and warnings are forwarded to the host compiler.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(roomsim PRIVATE
    -O3
    --compiler-options=-march=native,-Wall,-Wextra,-Wpedantic,${OpenMP_CXX_FLAGS}
)

target_link_libraries(roomsim PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    ${OpenMP_CXX_LIBRARIES}
    m
)
target_link_options(roomsim PRIVATE ${OpenMP_CXX_FLAGS})
