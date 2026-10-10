cmake_minimum_required(VERSION 3.24)
# Use the installed accelerators by default; cluster builds may override this
# with -DCMAKE_CUDA_ARCHITECTURES to cover other nodes.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(black_scholes_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source with nvcc; preserve the executable and filenames.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

# Do not use fast math or fused contractions: the benchmark relies on double
# precision and cancellation in the normal cumulative distribution function.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-O3,-ffp-contract=off,-Wall,-Wextra>)
target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
