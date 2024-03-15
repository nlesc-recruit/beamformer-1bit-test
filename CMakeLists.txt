cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(
  tensor-core-beamformer-test
  DESCRIPTION "1bit beamformer"
  VERSION 0.1
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)


# build beamformer
add_executable(bf src/bf.cu)
target_include_directories(bf PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                      ${CMAKE_SOURCE_DIR}/kernels
                                      ${CMAKE_SOURCE_DIR}/include)
# set_target_properties(bf PROPERTIES CUDA_ARCHITECTURES "native")  # requires cmake 3.24+
set_target_properties(bf PROPERTIES CUDA_ARCHITECTURES "86")  # A4000/A6000

install(TARGETS bf)
