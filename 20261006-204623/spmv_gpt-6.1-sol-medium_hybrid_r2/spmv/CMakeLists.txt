cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA; retain the executable and source names.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# nvcc must pass OpenMP options to the host compiler as well.
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
separate_arguments(OpenMP_HOST_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OpenMP_HOST_FLAGS)
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
