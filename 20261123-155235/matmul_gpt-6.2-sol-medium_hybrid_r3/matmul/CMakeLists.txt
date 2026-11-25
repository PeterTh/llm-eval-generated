cmake_minimum_required(VERSION 3.23)

# Build for the toolkit's supported GPU generations without requiring a GPU
# on the build host. The highest architecture also supplies forward-compatible PTX.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures")
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)
set_target_properties(matmul PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(matmul PRIVATE -O3 -Xcompiler=-O3,-Wall,-Wextra)
# FindOpenMP's CXX compile options are not applied to CUDA translation units.
separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_host_flags)
    target_compile_options(matmul PRIVATE "-Xcompiler=${flag}")
endforeach()
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cublas)
