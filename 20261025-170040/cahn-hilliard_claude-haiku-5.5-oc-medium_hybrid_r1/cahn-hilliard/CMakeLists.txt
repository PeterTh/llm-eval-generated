cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fat binary covering common accelerator generations. Set as a normal variable
# because CMake pre-populates the cache with the compiler's default (sm_52).
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The source contains CUDA kernels, so it is compiled by nvcc; host code is
# forwarded to the host compiler with OpenMP enabled.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# --fmad=false / -ffp-contract=off: strict IEEE evaluation of the expressions as
# written, so results are bit-reproducible across GPU and CPU and rank counts.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp,-ffp-contract=off>
)

target_link_libraries(cahn_hilliard PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
