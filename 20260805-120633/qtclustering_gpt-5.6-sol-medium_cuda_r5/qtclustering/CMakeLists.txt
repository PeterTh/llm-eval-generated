cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--compiler-options=-Wall,-Wextra>
)

# Build native device code when the caller did not select an architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET qtclustering PROPERTY CUDA_ARCHITECTURES native)
endif()
set_property(TARGET qtclustering PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(qtclustering m)
