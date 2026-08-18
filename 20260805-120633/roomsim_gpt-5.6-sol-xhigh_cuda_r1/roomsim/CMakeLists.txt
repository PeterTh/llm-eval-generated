cmake_minimum_required(VERSION 3.24)

# Set this before CUDA compiler detection; otherwise nvcc's conservative default
# architecture becomes a populated cache value before the project can select the
# GPU actually running the benchmark.  A command-line override is still honored.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# nvcc 12.0 predates some newer host compilers, while the generated host code
# remains compatible with them.  Device math keeps IEEE division/sqrt behavior;
# the hot Monte Carlo path selects fast intrinsics explicitly where safe.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --allow-unsupported-compiler
        --extra-device-vectorization
        -Xcompiler=-O3,-march=native,-Wall,-Wextra
    >
)

target_link_libraries(roomsim PRIVATE m)
