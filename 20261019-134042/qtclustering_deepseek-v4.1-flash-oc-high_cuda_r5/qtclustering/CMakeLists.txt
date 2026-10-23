cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

# The benchmark uses CUDA throughout; compile the single translation unit
# with NVCC even though it keeps the original .cpp name.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    # Build for the GPU(s) present on the machine doing the build.
    set_target_properties(qtclustering PROPERTIES CUDA_ARCHITECTURES native)
endif()

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Wall -Wextra -Xcompiler=-march=native>)

# Link math library
target_link_libraries(qtclustering m)
