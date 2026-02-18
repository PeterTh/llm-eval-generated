cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
# Note: -march=native is not supported by nvcc for host code in all versions, 
# but we can try -Xcompiler -march=native
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(roomsim PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall,-Wextra,-Wpedantic>
    )
else()
    target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
endif()

# Link math library
target_link_libraries(roomsim m)
