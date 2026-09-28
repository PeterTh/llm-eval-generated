cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# The single source file contains CUDA device code (__global__ kernels), so
# compile it with nvcc even though it keeps the .cpp extension.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations. OpenMP and host warning/optimization flags must be
# forwarded to the underlying host compiler via -Xcompiler since the file is
# compiled as CUDA.
target_compile_options(floydwarshall PRIVATE
    -O3
    -Xcompiler=-O3,-Wall,-Wextra,-fopenmp,-march=native
)

target_link_libraries(floydwarshall PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
