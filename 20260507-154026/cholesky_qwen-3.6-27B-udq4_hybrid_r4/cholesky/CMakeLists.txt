cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Host compilation flags
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# CUDA compilation flags
set_target_properties(cholesky PROPERTIES CUDA_ARCHITECTURES "60;70;75;80")
target_compile_options(cholesky PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")

# Link libraries
target_link_libraries(cholesky m ${MPI_CXX_LIBRARIES} cublas cuda ${OpenMP_CXX_LIBRARIES})
target_compile_options(cholesky PRIVATE ${OpenMP_CXX_FLAGS})

# Include directories
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})
