cmake_minimum_required(VERSION 3.20)

project(qtclustering LANGUAGES CXX CUDA)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(qtclustering PRIVATE
    -O3
    -Xcompiler=-march=native,-Wall,-Wextra
)
