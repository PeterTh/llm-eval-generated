cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the single source file as CUDA so it may contain both host and
# device (kernel) code without introducing a new file.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link math + CUDA runtime libraries
target_link_libraries(unstructured m)
