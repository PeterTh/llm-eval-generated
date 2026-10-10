cmake_minimum_required(VERSION 3.20)

# Callers can select the cluster GPU architectures with CMAKE_CUDA_ARCHITECTURES.
# Include PTX for forward compatibility with newer accelerators.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source filename and executable name.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
# OpenMP's imported target supplies host libraries; nvcc also needs its flags
# explicitly forwarded to the host compiler when compiling a CUDA source.
separate_arguments(FLOYDWARSHALL_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS FLOYDWARSHALL_OPENMP_FLAGS)
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
