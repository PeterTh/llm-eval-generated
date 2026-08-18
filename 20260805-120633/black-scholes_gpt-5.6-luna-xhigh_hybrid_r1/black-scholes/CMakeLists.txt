cmake_minimum_required(VERSION 3.18)

# Allow site-specific architecture selection with -DCMAKE_CUDA_ARCHITECTURES.
# Set the project default before enabling the CUDA language so CMake's compiler
# probe cannot replace it with its legacy fallback architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING
        "CUDA architectures for the black-scholes accelerator kernel")
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build the existing translation unit with nvcc; keeping the .cpp name
# preserves the benchmark's executable/source layout while enabling CUDA
# kernels alongside the MPI and OpenMP host code.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations for both the CUDA device and the GNU-compatible host
# compiler.  OpenMP is explicitly forwarded through nvcc because the source
# contains host-side OpenMP regions as well as device code.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp>
)
target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

set_target_properties(black_scholes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
