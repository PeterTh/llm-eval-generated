cmake_minimum_required(VERSION 3.18)

# Remember whether the user explicitly requested CUDA architectures, because
# project() may populate CMAKE_CUDA_ARCHITECTURES with a default value.
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_roomsim_user_cuda_arch TRUE)
else()
    set(_roomsim_user_cuda_arch FALSE)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pick the architecture of the GPU present on the build machine when possible,
# otherwise build a fat binary covering the common NVIDIA architectures.
if(NOT _roomsim_user_cuda_arch)
    set(_roomsim_cuda_arch "")
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _gpu_caps
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _smi_result)
        if(_smi_result EQUAL 0 AND _gpu_caps)
            string(REPLACE "\r" "" _gpu_caps "${_gpu_caps}")
            string(REPLACE "\n" ";" _gpu_caps "${_gpu_caps}")
            list(GET _gpu_caps 0 _gpu_cap)
            string(REPLACE "." "" _gpu_arch "${_gpu_cap}")
            if(_gpu_arch MATCHES "^[0-9]+$")
                set(_roomsim_cuda_arch "${_gpu_arch}")
            endif()
        endif()
    endif()

    if(_roomsim_cuda_arch)
        set(CMAKE_CUDA_ARCHITECTURES "${_roomsim_cuda_arch}" CACHE STRING
            "CUDA architectures to compile for" FORCE)
    else()
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING
            "CUDA architectures to compile for" FORCE)
    endif()
endif()

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-march=native>)

# Link math library
target_link_libraries(roomsim m)
