cmake_minimum_required(VERSION 3.20)
project(nbody_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(nbody PRIVATE -O3)

# The CUDA translation unit also contains the OpenMP host loops.
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(nbody PRIVATE --fmad=false)
    foreach(flag IN LISTS openmp_flags)
        target_compile_options(nbody PRIVATE "-Xcompiler=${flag}")
    endforeach()
else()
    target_compile_options(nbody PRIVATE -ffp-contract=off ${openmp_flags})
endif()
