cmake_minimum_required(VERSION 3.20)
project(floydwarshall_benchmark LANGUAGES CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(default_cuda_architectures ON)
endif()
enable_language(CUDA)
if(default_cuda_architectures AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    # Include native code for common cluster GPUs and PTX for newer targets.
    set(cuda_architectures 75-real 80-real 80-virtual)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
        list(PREPEND cuda_architectures 70-real)
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        list(APPEND cuda_architectures 90-real)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${cuda_architectures}" CACHE STRING
        "CUDA target architectures" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The existing source file contains both the host driver and the CUDA kernels.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(floydwarshall PRIVATE
        -O3 -Xcompiler=-O3,-Wall,-Wextra)
    foreach(flag IN LISTS openmp_flags)
        target_compile_options(floydwarshall PRIVATE "-Xcompiler=${flag}")
    endforeach()
else()
    target_compile_options(floydwarshall PRIVATE -O3 ${openmp_flags})
endif()
target_compile_definitions(floydwarshall PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
