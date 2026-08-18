cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Emit native Ampere machine code plus embedded PTX for forward compatibility.
# Callers can still override this on the CMake command line for another target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Keep host diagnostics while enabling CUDA's highest standard optimization
# level and device-side vectorization.  Fast math does not weaken the double
# precision add/multiply operations used by the SpMV kernel.
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extra-device-vectorization -Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(spmv m)
