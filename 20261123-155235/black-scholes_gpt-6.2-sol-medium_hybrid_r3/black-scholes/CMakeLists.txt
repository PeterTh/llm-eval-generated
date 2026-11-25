cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# This translation unit contains both CUDA kernels and OpenMP host code.
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_flags)
    target_compile_options(black_scholes PRIVATE "-Xcompiler=${flag}")
endforeach()
target_compile_options(black_scholes PRIVATE -O3 --fmad=false -Xcompiler=-Wall,-Wextra)
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
