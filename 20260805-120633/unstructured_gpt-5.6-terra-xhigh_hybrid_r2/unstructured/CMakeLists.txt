cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build native SASS by default for the accelerator on which the benchmark is
# configured.  This target-level setting deliberately overrides CMake's often
# conservative compiler default.  Sites cross-compiling can pass, for example,
# -DUNSTRUCTURED_CUDA_ARCHITECTURES=80-real instead.
set(UNSTRUCTURED_CUDA_ARCHITECTURES native CACHE STRING
    "CUDA architectures for the unstructured benchmark")

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    CUDA_ARCHITECTURES "${UNSTRUCTURED_CUDA_ARCHITECTURES}"
)

# CUDA owns this translation unit, so forward CPU optimization and OpenMP flags
# explicitly to its host compiler as well as optimizing device code.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(unstructured PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
