cmake_minimum_required(VERSION 3.18)

# Native code for Volta/Ampere and PTX for newer accelerators. Cluster builds
# can override this with -DCMAKE_CUDA_ARCHITECTURES=<their GPU architectures>.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "CUDA architectures")
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI 3 REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit 11 REQUIRED)

# Keep the existing source filename and executable name; nvcc compiles this TU.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
# Only MPI's C API is used; suppress its deprecated C++ bindings.
target_compile_definitions(floydwarshall PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)

target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)

# FindOpenMP's compile options apply to CXX sources; explicitly forward them to
# nvcc's host compiler too. Its imported target supplies the runtime libraries.
separate_arguments(FLOYDWARSHALL_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS FLOYDWARSHALL_OPENMP_FLAGS)
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
