cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename and executable name.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
