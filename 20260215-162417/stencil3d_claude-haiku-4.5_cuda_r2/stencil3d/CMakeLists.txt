cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(stencil3d stencil3d.cu)

target_link_libraries(stencil3d ${CUDA_LIBRARIES} m)

target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_70")
