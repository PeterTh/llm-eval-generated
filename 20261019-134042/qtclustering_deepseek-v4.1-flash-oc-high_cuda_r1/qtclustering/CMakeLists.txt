cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMake defaults CUDA architectures to 52, which would force PTX JIT and
# leave most of the GPU's capabilities unused.  Target the build machine's
# native architecture instead, unless the user explicitly requested another
# architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
   CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR
   CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
endif()

# The benchmark is kept in a .cpp file; compile it with nvcc instead of
# introducing additional source files.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
)

# Link math library
target_link_libraries(qtclustering m)
