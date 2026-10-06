cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Single source with host + device code: compile it as CUDA
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(qtclustering PROPERTIES LINKER_LANGUAGE CXX)

# Host code keeps the original flags (incl. GNU-mode FP contraction, so data
# generation and host-side distances are bit-identical to the original build).
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-ffp-contract=fast,${OpenMP_CXX_FLAGS}>)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(qtclustering PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link math library, MPI, OpenMP and CUDA runtime
target_link_libraries(qtclustering PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
