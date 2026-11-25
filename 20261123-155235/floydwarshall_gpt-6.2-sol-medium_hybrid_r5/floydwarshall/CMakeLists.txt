cmake_minimum_required(VERSION 3.20)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The existing source contains both the CUDA kernels and the MPI/OpenMP host code.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
target_compile_options(floydwarshall PRIVATE -O3)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(floydwarshall PRIVATE -Xcompiler=-O3,-Wall,-Wextra)
    foreach(flag IN LISTS openmp_flags)
        target_compile_options(floydwarshall PRIVATE "-Xcompiler=${flag}")
    endforeach()
else()
    target_compile_options(floydwarshall PRIVATE -Wall -Wextra ${openmp_flags})
endif()
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
