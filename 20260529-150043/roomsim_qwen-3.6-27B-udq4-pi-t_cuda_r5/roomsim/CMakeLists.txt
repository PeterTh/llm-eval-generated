cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture
execute_process(COMMAND nvidia-smi --query-gpu=compute_major,compute_minor --format=csv,noheader --id=0
                OUTPUT_VARIABLE GPU_COMPUTE
                RESULT_VARIABLE GPU_RESULT
                ERROR_QUIET)

if(GPU_RESULT EQUAL 0)
    string(REPLACE "," "." GPU_ARCH_RAW "${GPU_COMPUTE}")
    string(STRIP "${GPU_ARCH_RAW}" GPU_ARCH_RAW)
    string(REPLACE "." "" GPU_ARCH_RAW "${GPU_ARCH_RAW}")
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH_RAW})
else()
    set(CMAKE_CUDA_ARCHITECTURES 86)  # Default to Ampere (RTX 3090)
endif()

add_executable(roomsim roomsim.cpp)

# Force CUDA compilation for this source file
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-Wno-unused-parameter)
set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(roomsim m)
