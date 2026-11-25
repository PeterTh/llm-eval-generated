cmake_minimum_required(VERSION 3.20)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The benchmark's existing source contains both CUDA kernels and host code.
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

separate_arguments(OpenMP_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OpenMP_host_flags)
    target_compile_options(stencil3d PRIVATE "-Xcompiler=${flag}")
endforeach()
target_compile_options(stencil3d PRIVATE -O3 -Xcompiler=-O3,-Wall,-Wextra)
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
