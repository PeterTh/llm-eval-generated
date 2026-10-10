cmake_minimum_required(VERSION 3.24)

# Override this cache variable when building for a different cluster GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
target_compile_definitions(floydwarshall PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

# Explicitly forward OpenMP flags to nvcc's host compiler.
separate_arguments(OPENMP_HOST_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OPENMP_HOST_FLAGS)
    target_compile_options(floydwarshall PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(floydwarshall PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>")

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
