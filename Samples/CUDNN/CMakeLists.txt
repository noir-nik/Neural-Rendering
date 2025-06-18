cmake_minimum_required(VERSION 3.30)

if(CMAKE_VERSION VERSION_LESS "4.0.0") 
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
else()
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "a9e1cf81-9932-4810-974b-6eccaf14e457")
endif()

project(cu-test LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_MODULE_STD 1)

add_executable(${PROJECT_NAME}
	main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PRIVATE
	cudnn
	cudart
	cuda
)


target_link_directories(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})


