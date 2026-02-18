cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for broader compatibility before enable_language
set(CMAKE_CUDA_ARCHITECTURES "50;60;70;80")

add_executable(cahn_hilliard cahn_hilliard.cu)

# Set CUDA properties
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

# Include CUDA headers
target_include_directories(cahn_hilliard PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library and CUDA runtime
target_link_libraries(cahn_hilliard m)
