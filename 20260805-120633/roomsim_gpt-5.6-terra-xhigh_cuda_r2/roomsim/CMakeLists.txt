cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# CUDA is required: all compute phases run on the selected GPU and retain their
# working matrices there until the final host result is needed.
find_package(CUDAToolkit REQUIRED)

# Keep IEEE-like math for equivalent simulation semantics while enabling the
# compiler's normal high-performance GPU vectorization and host optimization.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xcompiler=-march=native,-Wall,-Wextra>
)

target_link_libraries(roomsim PRIVATE CUDA::cudart m)
