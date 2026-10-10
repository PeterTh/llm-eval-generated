cmake_minimum_required(VERSION 3.20)

# Include native code for common cluster GPUs and PTX for newer devices.
# A particular cluster can override this with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
endif()
project(unstructured_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing file as CUDA; keep the executable and source names.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)
set_target_properties(unstructured PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

string(REPLACE " " "," OPENMP_HOST_FLAGS "${OpenMP_CXX_FLAGS}")
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=${OPENMP_HOST_FLAGS},-Wall,-Wextra,-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-ffp-contract=off>)
target_link_libraries(unstructured PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
