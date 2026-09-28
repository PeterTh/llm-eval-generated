cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the single source file as CUDA so it can contain __global__ kernels
# alongside the MPI/OpenMP host code (hybrid MPI + OpenMP + CUDA).
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations. Host-side flags (OpenMP, march=native, warnings) are
# forwarded to the host compiler via -Xcompiler since nvcc drives compilation.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra>
)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# -fopenmp must also be present at link time (CMake links CUDA executables with the host compiler).
target_link_options(unstructured PRIVATE -fopenmp)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX m)
