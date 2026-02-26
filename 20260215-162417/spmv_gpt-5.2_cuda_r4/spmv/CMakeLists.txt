cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

# CUDA architectures (multi-arch fatbin for portability + good performance).
# CUDA 12 supports these, and the runtime will pick the best match.
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
set(CMAKE_CUDA_ARCHITECTURES 52;60;70;75;80;86;89;90)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep CUDA language at a widely-supported standard; spmv.cpp is compiled with NVCC.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

# Enable optimizations
target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math + CUDA runtime
target_link_libraries(spmv PRIVATE m CUDA::cudart)
