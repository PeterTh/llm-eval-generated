cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

# Prefer a concrete GPU architecture for best performance when supported
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile the existing source with NVCC to enable CUDA kernels
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp -Xcompiler=-march=native>
)

# Link libraries
target_link_libraries(roomsim PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
