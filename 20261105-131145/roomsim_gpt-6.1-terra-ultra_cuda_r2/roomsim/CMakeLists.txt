cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA; allow callers to override architectures.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-march=native>)
target_link_libraries(roomsim PRIVATE m)
