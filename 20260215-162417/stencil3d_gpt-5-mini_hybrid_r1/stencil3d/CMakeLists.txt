cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(stencil3d stencil3d.cpp)

# Tell CMake to compile this source with the CUDA compiler so kernels in .cpp are supported
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# Define visibility macros
target_compile_definitions(stencil3d PRIVATE USE_MPI USE_OPENMP USE_CUDA)
