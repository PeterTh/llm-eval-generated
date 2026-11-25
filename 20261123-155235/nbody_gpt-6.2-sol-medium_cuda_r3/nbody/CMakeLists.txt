cmake_minimum_required(VERSION 3.20)
project(nbody_benchmark LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
target_compile_options(nbody PRIVATE -O3 --fmad=false -Xcompiler=-Wall,-Wextra)
