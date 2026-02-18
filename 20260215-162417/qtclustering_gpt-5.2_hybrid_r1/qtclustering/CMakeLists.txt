cmake_minimum_required(VERSION 3.10)

# Let CMake pick the GPU architecture of the build machine (required by some CMake versions).
# Must be set before enabling the CUDA language (i.e., before project()).
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

enable_language(CUDA)

add_executable(qtclustering qtclustering.cpp)

# Compile the existing .cpp as CUDA to enable CUDA kernels without adding new files.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

# Link math library + MPI + OpenMP
target_link_libraries(qtclustering PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Ensure separable compilation is off (single TU) and keep host code performant.
set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
