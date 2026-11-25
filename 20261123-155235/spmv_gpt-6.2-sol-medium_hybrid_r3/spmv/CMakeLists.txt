cmake_minimum_required(VERSION 3.20)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

separate_arguments(omp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS omp_host_flags)
    target_compile_options(spmv PRIVATE "-Xcompiler=${flag}")
endforeach()
target_compile_options(spmv PRIVATE -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra)
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
