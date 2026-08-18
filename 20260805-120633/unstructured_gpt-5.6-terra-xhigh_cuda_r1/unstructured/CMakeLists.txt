cmake_minimum_required(VERSION 3.24)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build a native SASS image for the GPU that is compiling this benchmark.  The
# simulation is CUDA-only; there is intentionally no CPU execution fallback.
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Keep device code fully optimized while passing the same useful warnings and
# native host optimization through nvcc to the host compiler.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-march=native>
)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(unstructured PRIVATE CUDA::cudart m)
