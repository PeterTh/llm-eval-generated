cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES 60)
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler="-Wall -Wextra"> $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)

# Link math library
target_link_libraries(floydwarshall m)
