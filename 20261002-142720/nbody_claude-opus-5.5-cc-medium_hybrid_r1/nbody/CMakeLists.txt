cmake_minimum_required(VERSION 3.24)

# Build for the GPU(s) present on the build machine unless overridden
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(nbody nbody.cpp)

# The single source contains CUDA kernels: compile it with nvcc
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host flags forwarded to the host compiler)
target_compile_options(nbody PRIVATE
    -O3
    "SHELL:-Xcompiler -O3"
    "SHELL:-Xcompiler -march=native"
    "SHELL:-Xcompiler -Wall"
    "SHELL:-Xcompiler -Wextra"
    "SHELL:-Xcompiler ${OpenMP_CXX_FLAGS}"
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
# Only the MPI C API is used
target_compile_definitions(nbody PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(nbody PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
