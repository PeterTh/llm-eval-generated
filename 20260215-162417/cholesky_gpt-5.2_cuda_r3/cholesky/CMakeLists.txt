cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Compile this TU with NVCC so CUDA kernels are enabled without adding new files.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET cholesky PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)

# Prefer building for the locally-installed GPU when supported.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  set_property(TARGET cholesky PROPERTY CUDA_ARCHITECTURES native)
endif()

# Link math library
target_link_libraries(cholesky m)
