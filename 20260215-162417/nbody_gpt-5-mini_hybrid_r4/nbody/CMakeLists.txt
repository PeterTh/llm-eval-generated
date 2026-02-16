cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(nbody nbody.cpp)
# Tell CMake to compile the .cpp as CUDA so CUDA kernels can be in the file
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Optimization flags for host and device
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler "-march=native" -G0>
)

# Link libraries
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# Expose MPI include dirs if needed
if(MPI_CXX_COMPILE_OPTIONS)
    target_compile_options(nbody PRIVATE ${MPI_CXX_COMPILE_OPTIONS})
endif()
if(MPI_CXX_INCLUDE_PATH)
    target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()
