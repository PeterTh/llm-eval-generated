cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile the existing .cpp as a CUDA translation unit (no new files)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Optimization flags
target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  # For CUDA, pass tuning flags both to NVCC and the host compiler.
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options=-O3,-march=native,-fopenmp>
)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Prefer building for the current GPU when supported
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  set_property(TARGET qtclustering PROPERTY CUDA_ARCHITECTURES native)
endif()
