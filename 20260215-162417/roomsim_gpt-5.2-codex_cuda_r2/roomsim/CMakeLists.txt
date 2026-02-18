cmake_minimum_required(VERSION 3.10)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic>
)

# Link math library
target_link_libraries(roomsim m)
