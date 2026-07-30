cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

# C++17 for better CUDA compatibility (nvcc 12.0 + C++20 has issues)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Force CUDA compilation for the .cpp source
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings with relaxed constexpr for device code access
target_compile_options(roomsim PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Set CUDA architecture for RTX 3090 (compute 8.6)
set_target_properties(roomsim PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
)

# Link math library and CUDA runtime
target_link_libraries(roomsim m cudart)
