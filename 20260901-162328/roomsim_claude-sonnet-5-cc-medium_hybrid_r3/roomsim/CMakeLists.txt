cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90 CACHE STRING "CUDA architectures to target")

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile the single source file as CUDA so it can host MPI calls, OpenMP
# pragmas, and __global__ kernels together (hybrid MPI + OpenMP + CUDA).
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings. OpenMP must be forwarded explicitly to
# the host compiler via -Xcompiler since nvcc does not consume -fopenmp itself.
target_compile_options(roomsim PRIVATE
    -O3 -Xcompiler=-march=native,-fopenmp,-Wall,-Wextra
)

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::curand m)
