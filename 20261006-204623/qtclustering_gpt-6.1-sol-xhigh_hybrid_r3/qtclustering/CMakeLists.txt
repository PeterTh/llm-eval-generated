cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename while compiling its kernels with nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

target_compile_definitions(qtclustering PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra,-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-ffp-contract=off>)

# OpenMP's imported target supplies the host runtime; nvcc also needs the
# compiler flags explicitly forwarded to its host compiler.
separate_arguments(QT_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS QT_OPENMP_FLAGS)
    target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
