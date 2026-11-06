cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the existing source and executable names while compiling the CUDA kernels.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# OpenMP's CXX target supplies its compile flags only to CXX compilation;
# nvcc must forward them explicitly to its host compiler for this CUDA source.
separate_arguments(SPMV_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS SPMV_OPENMP_FLAGS)
    target_compile_options(spmv PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(spmv PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>")
