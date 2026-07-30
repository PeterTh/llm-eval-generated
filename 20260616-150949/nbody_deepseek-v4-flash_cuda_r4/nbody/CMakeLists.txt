cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# nvcc -O3 optimizes both host and device code
target_compile_options(nbody PRIVATE -O3)

# Link math library (still needed for host-side validation code)
target_link_libraries(nbody m)
