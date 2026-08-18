cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)
# Keep the established .cpp source and executable name while compiling the
# translation unit with nvcc so the CUDA kernel is always part of the binary.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
                                      $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp;-Xcompiler=-Wall,-Wextra>)
target_compile_features(qtclustering PRIVATE cxx_std_20 cuda_std_17)
set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
