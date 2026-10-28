cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on this machine
set(CMAKE_CUDA_ARCHITECTURES native)

# roomsim.cpp is compiled as CUDA source (nvcc handles host and device code)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings (no fast-math: results must match the sequential semantics)
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options=-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(roomsim m)
