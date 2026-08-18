cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source name while compiling it with NVCC.  CUDA is a
# required language for this benchmark: there is intentionally no CPU path.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Generate code for the GPU present at configure time, rather than relying on
# PTX JIT at benchmark startup.
set_target_properties(roomsim PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION OFF
)

# Optimize both device code and the NVCC-host-compiled setup code.  Fast math
# is deliberately not enabled so the floating-point model remains faithful to
# the original implementation.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(roomsim PRIVATE CUDA::cudart m)
