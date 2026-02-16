cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60)

add_executable(floydwarshall floydwarshall.cpp)
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> -O3 -Wall -Wextra)
target_link_libraries(floydwarshall m)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -Wall -Wextra)

# Link math library and CUDA
if(CUDA_FOUND)
    target_link_libraries(floydwarshall m ${CUDA_LIBRARIES})
endif()
