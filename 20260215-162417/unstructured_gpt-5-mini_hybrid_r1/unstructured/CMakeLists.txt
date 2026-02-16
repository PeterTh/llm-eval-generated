cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

# Detect NVCC to enable CUDA support if available
find_program(NVCC_EXECUTABLE nvcc)
if(NVCC_EXECUTABLE)
    message(STATUS "nvcc found: enabling CUDA support")
    enable_language(CUDA)
    set(HAVE_CUDA ON)
else()
    message(STATUS "nvcc not found: building without CUDA kernels")
    set(HAVE_CUDA OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# If CUDA is available, compile the source as CUDA to enable kernels
if(HAVE_CUDA)
    set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
# Expose parallelization defines
if(HAVE_CUDA)
    target_compile_definitions(unstructured PRIVATE USE_MPI USE_OPENMP USE_CUDA)
else()
    target_compile_definitions(unstructured PRIVATE USE_MPI USE_OPENMP)
endif()

# Link math library and MPI/OpenMP
if(HAVE_CUDA)
    # Link against CUDA if found
    target_link_libraries(unstructured PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_LIBRARIES})
else()
    target_link_libraries(unstructured PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()
