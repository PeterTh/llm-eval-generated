cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The existing source contains both the host implementation and CUDA kernels.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-march=native,-Wall,-Wextra>)
# OpenMP's imported target supplies the link flags, but its CXX-only compile
# options do not apply to host code compiled through nvcc.
separate_arguments(_openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(_flag IN LISTS _openmp_host_flags)
    target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>)
endforeach()
