cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the available GPU by default; callers may select deployment targets
# with -DCMAKE_CUDA_ARCHITECTURES=... when cross-compiling.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

# Keep the existing source filename and executable name.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Keep double-precision math, including division and sqrt. Disabling contraction
# preserves the original expression rounding; do not enable CUDA fast math.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>
)
target_link_libraries(black_scholes PRIVATE m)
