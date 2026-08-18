cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

enable_language(CUDA)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(unstructured PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)

# Enable optimizations
target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-fopenmp> $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)

# Link math library
target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m mpi_cxx mpi)
# nvcc does not consume the include flags embedded in the MPI wrapper target.
# Obtain the wrapper's include directories and apply them to the CUDA compile.
execute_process(COMMAND ${MPI_CXX_COMPILER} --showme:incdirs
                OUTPUT_VARIABLE MPI_WRAPPER_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(MPI_WRAPPER_INCLUDE_DIRS)
target_include_directories(unstructured PRIVATE ${MPI_WRAPPER_INCLUDE_DIRS})
execute_process(COMMAND ${MPI_CXX_COMPILER} --showme:link
                OUTPUT_VARIABLE MPI_WRAPPER_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(MPI_WRAPPER_LINK_FLAGS)
target_link_options(unstructured PRIVATE ${MPI_WRAPPER_LINK_FLAGS})
