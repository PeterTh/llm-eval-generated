cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find dependencies ----
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture; fall back to common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

add_executable(stencil3d stencil3d.cu)

# ---- Optimisation flags ----
target_compile_options(stencil3d PRIVATE
    -O3 -march=native -Wall -Wextra
    --expt-extended-lambda
    -Xcompiler=-fopenmp
)

# ---- Link ----
target_link_libraries(stencil3d
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
