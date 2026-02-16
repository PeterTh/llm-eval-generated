cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

# Enable MPI, OpenMP and CUDA detection
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# Prefer detecting nvcc to determine CUDA availability to avoid CMake's deprecated FindCUDA behavior
find_program(NVCC_EXECUTABLE nvcc)
if(NVCC_EXECUTABLE)
  set(CUDA_FOUND TRUE)
else()
  set(CUDA_FOUND FALSE)
  message(STATUS "CUDA not found; building CPU+OpenMP+MPI only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations and parallel flags
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
  target_compile_options(cahn_hilliard PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(cahn_hilliard PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MPI_CXX_FOUND)
  target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_PATH})
  # prefer imported target if available
  if(TARGET MPI::MPI_CXX)
    target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX)
  else()
    target_link_libraries(cahn_hilliard PRIVATE ${MPI_CXX_LIBRARIES})
  endif()
  target_compile_definitions(cahn_hilliard PRIVATE USE_MPI)
endif()

if(CUDA_FOUND)
  enable_language(CUDA)
  message(STATUS "Building with CUDA support")
  target_compile_definitions(cahn_hilliard PRIVATE USE_CUDA)
  # compile as CUDA-capable host code
  set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Link math library
target_link_libraries(cahn_hilliard PRIVATE m)
