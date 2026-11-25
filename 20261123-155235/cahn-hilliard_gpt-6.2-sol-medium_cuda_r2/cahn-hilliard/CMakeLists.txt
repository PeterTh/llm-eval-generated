cmake_minimum_required(VERSION 3.20)
project(cahn_hilliard_benchmark LANGUAGES CUDA)

# Keep the existing source filename and compile it as CUDA.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
# The kernels explicitly fuse operations where the serial benchmark does.
target_compile_options(cahn_hilliard PRIVATE -O3 --fmad=false -Xcompiler=-Wall,-Wextra)
