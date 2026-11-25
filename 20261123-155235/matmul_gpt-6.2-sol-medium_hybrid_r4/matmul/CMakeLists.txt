cmake_minimum_required(VERSION 3.20)

# Ship native code for the major architectures supported by the toolkit, plus PTX.
# CUDAARCHS or -DCMAKE_CUDA_ARCHITECTURES can target a specific cluster.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23
        AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
list(JOIN openmp_host_flags "," openmp_host_flags)
target_compile_options(matmul PRIVATE -O3 --fmad=false
    "-Xcompiler=${openmp_host_flags},-Wall,-Wextra")
target_compile_definitions(matmul PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
