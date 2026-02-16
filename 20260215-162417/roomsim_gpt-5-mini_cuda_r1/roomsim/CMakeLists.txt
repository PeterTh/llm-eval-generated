cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(NVCC nvcc)
if(NOT NVCC)
  message(FATAL_ERROR "nvcc not found; CUDA toolkit required")
endif()

# Build the roomsim executable using nvcc for CUDA acceleration (unconditional)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/roomsim
  COMMAND ${NVCC} -O3 -std=c++20 ${CMAKE_CURRENT_SOURCE_DIR}/roomsim.cpp -o ${CMAKE_CURRENT_SOURCE_DIR}/roomsim -I${CMAKE_CURRENT_SOURCE_DIR}/../common -Xcompiler "-Wall -Wextra -Wpedantic -march=native" -lm
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/roomsim.cpp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(roomsim ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/roomsim)
