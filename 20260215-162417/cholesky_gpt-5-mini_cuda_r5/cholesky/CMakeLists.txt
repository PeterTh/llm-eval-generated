cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build the single source file with the CUDA compiler
add_executable(cholesky cholesky.cpp)
# Tell CMake to treat the .cpp as CUDA source so nvcc compiles it
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
if (CMAKE_CUDA_COMPILER)
  target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=compute_70 -Xfatbin -compress-all>)
endif()
# Host compile options
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link CUDA runtime and math library
find_package(CUDA QUIET)
if (CUDA_FOUND)
  target_link_libraries(cholesky PRIVATE ${CUDA_CUDART_LIBRARY} m)
else()
  target_link_libraries(cholesky PRIVATE m)
endif()
