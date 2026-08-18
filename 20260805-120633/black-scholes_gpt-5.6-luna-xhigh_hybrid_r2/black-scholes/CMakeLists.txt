cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the default executable portable across the common accelerator
# generations. Users building on a fixed cluster can override this with
# -DCMAKE_CUDA_ARCHITECTURES=<arch> (for example, 86 for an RTX 3090).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86;89;90)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The benchmark intentionally remains a single existing source file. Compile
# it as CUDA so the host MPI/OpenMP driver and the device pricing kernel are
# built into the required black_scholes executable.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-fopenmp;-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)
target_compile_definitions(black_scholes PRIVATE BS_HYBRID_MPI_OPENMP_CUDA=1)

# MPI supplies distributed work, OpenMP supplies host-side parallelism, and
# the CUDA runtime launches the accelerator kernel. No additional runtime or
# third-party dependency is introduced.
target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)

set_target_properties(black_scholes PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
