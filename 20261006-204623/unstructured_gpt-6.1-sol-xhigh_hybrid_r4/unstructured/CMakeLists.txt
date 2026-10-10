cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the source and executable names; nvcc compiles the device kernels in .cpp.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# OpenMP's imported target supplies host linking, but its CXX-only compile flags
# must also be forwarded explicitly through nvcc to its host compiler.
separate_arguments(OPENMP_HOST_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OPENMP_HOST_FLAGS)
    target_compile_options(unstructured PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
endforeach()
target_compile_options(unstructured PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-ffp-contract=off>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra,-ffp-contract=off>")

# All three parallel runtimes are required, including for one-rank runs.
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
