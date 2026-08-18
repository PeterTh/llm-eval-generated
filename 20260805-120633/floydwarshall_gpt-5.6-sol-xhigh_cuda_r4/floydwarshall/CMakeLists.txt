cmake_minimum_required(VERSION 3.24)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU on which the benchmark is configured.  This avoids
# shipping generic PTX through the driver's JIT and lets nvcc use all of the
# target GPU's instructions.
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable host and device optimizations.  The CUDA hot path uses only integer
# arithmetic, so the fast-math mode cannot alter shortest-path results.
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;-Xcompiler=-Wall,-Wextra,-march=native>
)

# Link math library
target_link_libraries(floydwarshall m)
