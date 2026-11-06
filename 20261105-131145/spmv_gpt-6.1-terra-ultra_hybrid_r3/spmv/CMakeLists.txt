cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The CUDA implementation lives in the existing source file.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>)

# OpenMP's imported target supplies the runtime when linking, but its compile
# options apply to C++ sources only. Forward them through nvcc for this source.
separate_arguments(_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(_flag IN LISTS _openmp_flags)
    target_compile_options(spmv PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>")
endforeach()
