cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(roomsim roomsim.cpp)
target_link_libraries(roomsim m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda cudart)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60 -O3> $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link math library
target_link_libraries(roomsim m)
