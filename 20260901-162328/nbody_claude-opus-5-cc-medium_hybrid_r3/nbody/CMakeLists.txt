cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
  endif()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# nbody.cpp holds host and device code, selected by __CUDACC__.  The device
# half is compiled from a .cu symlink so that nvcc picks it up, while the host
# half goes through the regular C++ compiler - the host force loop needs the
# vectoriser of a plain C++ compilation.
set(NBODY_DEVICE_SRC ${CMAKE_CURRENT_BINARY_DIR}/nbody_device.cu)
file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/nbody.cpp ${NBODY_DEVICE_SRC} SYMBOLIC)

add_library(nbody_device OBJECT ${NBODY_DEVICE_SRC})
set_target_properties(nbody_device PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(nbody_device PRIVATE -O3)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(nbody PRIVATE nbody_device MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
