cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

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

# Use the same host compiler as for plain C++ so that host floating point code
# is generated exactly as by the reference build.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The benchmark lives in a single translation unit that contains both host and
# device code; compile it with the CUDA toolchain.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings (host flags are forwarded to the host compiler)
target_compile_options(roomsim PRIVATE
    -O3
    --expt-relaxed-constexpr
    --prec-div=true --prec-sqrt=true
    -Xcompiler=-O3,-march=native,-Wall,-Wextra
    -Xcompiler=${OpenMP_CXX_FLAGS}
)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_options(roomsim PRIVATE ${OpenMP_CXX_FLAGS})

# Link math, MPI, OpenMP and the CUDA runtime
target_link_libraries(roomsim PRIVATE m ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
