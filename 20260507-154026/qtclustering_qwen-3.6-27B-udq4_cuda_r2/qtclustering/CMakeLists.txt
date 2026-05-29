cmake_minimum_required(VERSION 3.10)

project(qtclustering CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES CUDA_STANDARD 17)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)
# Detect GPU compute capability for optimal codegen
execute_process(
    COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1"
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(GPU_COMPUTE_CAP)
    string(REPLACE "." "" GPU_CC "${GPU_COMPUTE_CAP}")
    message(STATUS "Detected GPU compute capability: ${GPU_COMPUTE_CAP}")
    set(CUDA_ARCH_FLAG "-arch=compute_${GPU_CC};-code=sm_${GPU_CC}")
else()
    set(CUDA_ARCH_FLAG "-arch=compute_80;-code=sm_80")
    message(STATUS "Could not detect GPU, defaulting to sm_80")
endif()
target_compile_options(qtclustering PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;${CUDA_ARCH_FLAG}>")

# Link math library
target_link_libraries(qtclustering m)
