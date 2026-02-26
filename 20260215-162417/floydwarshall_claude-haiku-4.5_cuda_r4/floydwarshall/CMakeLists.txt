cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_executable(floydwarshall floydwarshall.cu)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-march=native> $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "native")

# Link CUDA runtime and math library
target_link_libraries(floydwarshall PRIVATE m)
