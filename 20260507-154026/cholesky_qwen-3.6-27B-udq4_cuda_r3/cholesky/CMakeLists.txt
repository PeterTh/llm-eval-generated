cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES LINKER_LANGUAGE CUDA)

# Force cholesky.cpp to be compiled with nvcc
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable host optimizations
set(CMAKE_CUDA_ARCHITECTURES OFF)
target_compile_options(cholesky PRIVATE --expt-relaxed-constexpr -Xcompiler=-O3,-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(cholesky m)
