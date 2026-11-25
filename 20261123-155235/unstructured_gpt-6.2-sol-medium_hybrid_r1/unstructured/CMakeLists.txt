cmake_minimum_required(VERSION 3.20)
project(unstructured_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(unstructured PRIVATE -O3)

# nvcc must pass the OpenMP options to its host compiler.
separate_arguments(openmp_host_options NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(option IN LISTS openmp_host_options)
    target_compile_options(unstructured PRIVATE "-Xcompiler=${option}")
endforeach()
target_compile_definitions(unstructured PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
