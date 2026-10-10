cmake_minimum_required(VERSION 3.24)

# Override this cache variable when building for a different cluster GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX ON)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Query GPU buffer support when the MPI implementation provides the extension.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("
    #include <mpi.h>
    #if __has_include(<mpi-ext.h>)
    #include <mpi-ext.h>
    #endif
    int main() { return MPIX_Query_cuda_support(); }
" HAVE_MPI_CUDA_QUERY)
unset(CMAKE_REQUIRED_LIBRARIES)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_compile_definitions(cahn_hilliard PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
if(HAVE_MPI_CUDA_QUERY)
    target_compile_definitions(cahn_hilliard PRIVATE HAVE_MPI_CUDA_QUERY)
endif()

# Forward OpenMP to nvcc's host compiler and preserve floating-point semantics.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra,-ffp-contract=off>
)
separate_arguments(OpenMP_CXX_FLAGS_LIST NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OpenMP_CXX_FLAGS_LIST)
    target_compile_options(cahn_hilliard PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()
