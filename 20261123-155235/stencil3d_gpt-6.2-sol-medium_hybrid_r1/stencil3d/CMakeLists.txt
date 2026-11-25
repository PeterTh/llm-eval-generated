cmake_minimum_required(VERSION 3.23)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
project(stencil3d_benchmark LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Shared
    LINKER_LANGUAGE CXX)

# OpenMP's imported target supplies the runtime. Pass its C++ flags through nvcc.
separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_host_flags)
    target_compile_options(stencil3d PRIVATE "-Xcompiler=${flag}")
endforeach()
target_compile_options(stencil3d PRIVATE -O3 -Xcompiler=-Wall,-Wextra)
target_compile_definitions(stencil3d PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(stencil3d PRIVATE CUDA::cudart MPI::MPI_CXX OpenMP::OpenMP_CXX)
