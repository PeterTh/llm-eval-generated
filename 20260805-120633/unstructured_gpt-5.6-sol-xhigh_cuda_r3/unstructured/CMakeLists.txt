cmake_minimum_required(VERSION 3.24)

# Select the installed GPU before CUDA compiler detection supplies its generic
# default.  Callers can still override this cache variable explicitly.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build the existing source as CUDA without changing the benchmark's source or
# executable names.  "native" emits code specifically for the benchmark GPU.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable host and device optimizations.  CUDA's precise double intrinsics in
# the update kernel retain the reference operation ordering.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Link math library
target_link_libraries(unstructured m)
