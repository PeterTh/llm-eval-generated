cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set_target_properties(cahn_hilliard PROPERTIES LINKER_LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(cahn_hilliard m)
