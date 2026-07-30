cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings (passed to nvcc host compiler)
target_compile_options(roomsim PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wpedantic>
)

# Link math library
target_link_libraries(roomsim m)
