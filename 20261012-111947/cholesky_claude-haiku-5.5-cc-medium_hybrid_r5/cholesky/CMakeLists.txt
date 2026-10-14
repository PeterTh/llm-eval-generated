cmake_minimum_required(VERSION 3.24)

# Target common data-center and workstation GPUs unless overridden (Volta .. Hopper).
# Must be set before project(), which otherwise picks the nvcc default architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# GNU dialect on the host side so floating-point contraction matches the CPU-only build
set(CMAKE_CUDA_EXTENSIONS ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# cholesky.cpp holds CUDA kernels, so it is compiled by nvcc
add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-ffp-contract=fast
    -Xcompiler=-fopenmp
)

target_link_options(cholesky PRIVATE -fopenmp)

target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, CUDA runtime and math library
target_link_libraries(cholesky PRIVATE ${MPI_CXX_LIBRARIES} CUDA::cudart m)
