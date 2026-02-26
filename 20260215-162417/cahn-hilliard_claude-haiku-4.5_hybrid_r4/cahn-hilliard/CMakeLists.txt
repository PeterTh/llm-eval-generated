cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cu)

set_target_properties(cahn_hilliard PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70 75 80 86 90
)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native>
)

# Link libraries
target_link_libraries(cahn_hilliard 
    PRIVATE 
    m 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
)

target_include_directories(cahn_hilliard PRIVATE ${CUDA_INCLUDE_DIRS})
