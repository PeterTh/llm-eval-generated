cmake_minimum_required(VERSION 3.18)

# Keep the source filename and executable name. All three runtimes are required.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 CACHE STRING "CUDA architectures")
endif()
project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES LINKER_LANGUAGE CXX)

# OpenMP's imported target supplies host libraries, but its CXX compile flags
# must also be forwarded through nvcc for this CUDA translation unit.
separate_arguments(QT_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(QT_OPENMP_FLAG IN LISTS QT_OPENMP_FLAGS)
    target_compile_options(qtclustering PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${QT_OPENMP_FLAG}>")
endforeach()
target_compile_options(qtclustering PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>")
target_compile_definitions(qtclustering PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
