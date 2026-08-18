cmake_minimum_required(VERSION 3.24)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Build for the GPUs visible on the build node instead of CMake's old,
# conservative default architecture.
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA so it can contain the GPU kernel while
# retaining the original executable and source-file layout.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    # CUDA compilation owns this source; forward OpenMP to its host compiler
    # so the host initialization/validation loops are truly parallel.
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp>)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
