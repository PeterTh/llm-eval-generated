cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source/executable names while compiling the CUDA kernels.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

# MPI and OpenMP are mandatory; forward OpenMP flags to nvcc's host compiler.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra,-ffp-contract=off>)
separate_arguments(MATMUL_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS MATMUL_OPENMP_FLAGS)
    target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
