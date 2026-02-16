cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Detect nvcc (CUDA) toolchain; allow fallback when not present
find_program(NVCC_EXECUTABLE nvcc)
if(NVCC_EXECUTABLE)
  message(STATUS "NVCC found: ${NVCC_EXECUTABLE}, enabling CUDA build")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_definitions(-DUSE_CUDA=1)
else()
  message(WARNING "NVCC not found - building with CPU fallback (OpenMP + MPI only)")
  add_definitions(-DUSE_CUDA=0)
endif()

add_executable(matmul matmul.cpp)

# If CUDA enabled, compile with CUDA properties
if(NVCC_EXECUTABLE)
  set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo> $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-O3 -march=native -Wall -Wextra>)
else()
  target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link libraries: MPI, OpenMP and math
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
