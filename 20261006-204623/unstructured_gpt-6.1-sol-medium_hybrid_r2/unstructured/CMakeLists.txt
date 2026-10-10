cmake_minimum_required(VERSION 3.18)
# Override with -DCMAKE_CUDA_ARCHITECTURES for the target accelerator cluster.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80)
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the existing source file and executable name.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
# OpenMP's imported target does not propagate CXX compile flags to CUDA sources.
separate_arguments(OPENMP_HOST_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OPENMP_HOST_FLAGS)
    target_compile_options(unstructured PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(unstructured PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra,-ffp-contract=off>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-ffp-contract=off>")
