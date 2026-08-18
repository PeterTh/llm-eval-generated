cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark is intentionally CUDA-only.  The target hardware is an
# Ampere-class GPU; callers may override this cache value for another device.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(roomsim roomsim.cpp)

# Keep the existing source file and executable name while compiling the
# implementation as CUDA C++.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable aggressive host and device optimization.  No CPU fallback is linked.
target_compile_options(roomsim PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
    "$<$<COMPILE_LANGUAGE:CXX>:-march=native>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>"
)

# Link math library
target_link_libraries(roomsim PRIVATE m)
