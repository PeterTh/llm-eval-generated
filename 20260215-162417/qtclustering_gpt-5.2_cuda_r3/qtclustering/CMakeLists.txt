cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Build for the local GPU for best performance.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

# Build the existing .cpp as CUDA to unconditionally use GPU parallelism.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(qtclustering m)
