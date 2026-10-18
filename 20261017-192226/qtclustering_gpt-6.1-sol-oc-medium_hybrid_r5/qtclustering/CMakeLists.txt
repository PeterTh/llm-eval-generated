cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

# Preserve double-precision distance comparisons, including exact ties.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>
)
foreach(flag IN LISTS OpenMP_CXX_FLAGS)
    target_compile_options(qtclustering PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
