cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
project(qtclustering LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Match distance arithmetic to the original native host build.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-march=native")
check_cxx_source_compiles("
    #ifndef __FMA__
    #error Host FMA is unavailable
    #endif
    int main() {}
" QT_HOST_FMA)
unset(CMAKE_REQUIRED_FLAGS)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

separate_arguments(qt_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS qt_openmp_flags)
    target_compile_options(qtclustering PRIVATE "-Xcompiler=${flag}")
endforeach()
target_compile_options(qtclustering PRIVATE -O3 --fmad=false
    -Xcompiler=-march=native,-ffp-contract=fast,-Wall,-Wextra)
target_compile_definitions(qtclustering PRIVATE QT_HOST_FMA=$<BOOL:${QT_HOST_FMA}>
    OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
