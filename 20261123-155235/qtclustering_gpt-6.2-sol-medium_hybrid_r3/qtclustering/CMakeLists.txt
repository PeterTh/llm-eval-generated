cmake_minimum_required(VERSION 3.20)

project(qtclustering LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Compile the existing translation unit with nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
list(JOIN openmp_host_flags "," openmp_host_flags)
target_compile_options(qtclustering PRIVATE -O3
    -Xcompiler=-O3,-march=native,-Wall,-Wextra
    "-Xcompiler=${openmp_host_flags}")
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
