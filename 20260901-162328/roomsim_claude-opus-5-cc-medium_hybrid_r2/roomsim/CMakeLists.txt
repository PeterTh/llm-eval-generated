cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
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

# The single translation unit contains both host and device code
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings (host flags are forwarded through nvcc)
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(roomsim PRIVATE ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)
target_link_options(roomsim PRIVATE -fopenmp)
