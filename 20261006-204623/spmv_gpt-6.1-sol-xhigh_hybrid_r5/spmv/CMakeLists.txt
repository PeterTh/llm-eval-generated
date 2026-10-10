cmake_minimum_required(VERSION 3.18)

# Include PTX so the executable also runs on newer accelerator generations.
# A cluster can supply its own list with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80)
endif()
project(spmv_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit 11 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename while compiling its CUDA kernels with nvcc.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)
set_target_properties(spmv PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-Wall,-Wextra>
)

# OpenMP's imported target supplies the runtime, but its compiler flags are
# restricted to CXX. Forward those flags to nvcc's host compiler explicitly.
separate_arguments(SPMV_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS SPMV_OPENMP_FLAGS)
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
