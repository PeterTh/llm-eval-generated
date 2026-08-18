cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the established executable name while compiling the implementation as
# CUDA C++; MPI ranks select one accelerator each at runtime.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
get_target_property(MPI_INCLUDE_DIRS MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(black_scholes PRIVATE ${MPI_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_PATH})
# FindMPI's imported target is language-specific; CUDA needs the MPI wrapper's
# header flags explicitly because this source is compiled by nvcc.
execute_process(COMMAND ${MPI_CXX_COMPILER} --showme:compile
                OUTPUT_VARIABLE MPI_CUDA_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(MPI_CUDA_COMPILE_FLAGS NATIVE_COMMAND "${MPI_CUDA_COMPILE_FLAGS}")
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MPI_CUDA_COMPILE_FLAGS}>)
execute_process(COMMAND ${MPI_CXX_COMPILER} --showme:link
                OUTPUT_VARIABLE MPI_CUDA_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(MPI_CUDA_LINK_FLAGS NATIVE_COMMAND "${MPI_CUDA_LINK_FLAGS}")

# Enable optimizations
target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
                                          $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m ${MPI_CUDA_LINK_FLAGS})
