cmake_minimum_required(VERSION 3.10)

# Ensure CUDA architectures are not left empty on newer CMake/NVCC combos.
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Reasonable default fatbin set for portability; can be overridden by the user.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  endif()
endif()

# Compile the existing .cpp with nvcc (no new files, same executable name)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(cahn_hilliard m)
