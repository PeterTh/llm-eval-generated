cmake_minimum_required(VERSION 3.20)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Compile the existing source with NVCC without changing its name.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Forward the compiler-specific OpenMP flags to the CUDA host compiler.
separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
list(JOIN openmp_host_flags "," openmp_host_flags)
# Match the reference rounding of each flux operation.
target_compile_options(unstructured PRIVATE
    -O3 --fmad=false "-Xcompiler=${openmp_host_flags},-Wall,-Wextra")
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
