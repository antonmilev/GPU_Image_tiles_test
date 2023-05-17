cmake_minimum_required (VERSION 3.26)
project (test_filter LANGUAGES CUDA CXX)

include(CheckLanguage)
# The version number.
set (hello_cuda_VERSION_MAJOR 1)
set (hello_cuda_VERSION_MINOR 0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

message(CUDA_INCLUDE_DIRS="${CUDA_INCLUDE_DIRS}")
message(CUDA_LIBRARIES="${CUDA_LIBRARIES}")

# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")

get_filename_component(PARENT_DIR ../ ABSOLUTE)
message(PARENT_DIR="${PARENT_DIR}")

message(COMMON_DIR="${COMMON_DIR}")

if  (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")
  set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PARENT_DIR}/bin)

set(CMAKE_VERBOSE_MAKEFILE ON)
add_executable(test_filter 
                  Main.cpp  
				  convolution_gpu.cu
				  )

target_link_libraries(test_filter ${CUDA_LIBRARIES})



