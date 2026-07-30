cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target SM architecture: 86 = RTX 30xx (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Host compiler optimizations
set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations for both host and device code
target_compile_options(floydwarshall PRIVATE
    -O3
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)

# Link math library
target_link_libraries(floydwarshall m)
