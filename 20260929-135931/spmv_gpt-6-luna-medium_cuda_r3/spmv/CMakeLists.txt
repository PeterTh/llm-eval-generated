cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(spmv m)
