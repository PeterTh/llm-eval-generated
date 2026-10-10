cmake_minimum_required(VERSION 3.18)
# Tune for the available GPU unless the caller selects target architectures.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24 AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source and executable names; compile this file as CUDA.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Preserve ordered sums, native host arithmetic, and visibility tolerances.
# CUDA's default precise division/sqrt and non-FTZ behavior are intentional.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=true;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)
target_link_libraries(roomsim PRIVATE m)
