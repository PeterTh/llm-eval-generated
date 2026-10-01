cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

# Compile the existing source as CUDA without changing the executable name.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

# Keep double-precision distance comparisons and deterministic threshold ties.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
target_link_libraries(qtclustering PRIVATE m)
