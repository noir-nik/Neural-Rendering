cmake_minimum_required(VERSION 3.30)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")

project(NeuralGraphics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_MODULE_STD 1)

option(NG_BUILD_SAMPLES "Build samples" OFF)

# add_executable(${PROJECT_NAME})

add_library(${PROJECT_NAME} ${SOURCES})
add_library(NeuralGraphics::NeuralGraphics ALIAS NeuralGraphics)

find_package(Vulkan REQUIRED)
if( ${Vulkan_VERSION} VERSION_LESS "1.3.296" )
message( FATAL_ERROR "Minimum required Vulkan version for C++ modules is 1.3.296. Found ${Vulkan_VERSION}.")
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

set(VULKAN_HPP_DEFINITIONS
	VULKAN_HPP_NO_EXCEPTIONS
	VULKAN_HPP_RAII_NO_EXCEPTIONS
	VULKAN_HPP_NO_SMART_HANDLE
	VULKAN_HPP_NO_CONSTRUCTORS
	VULKAN_HPP_NO_UNION_CONSTRUCTORS
	VULKAN_HPP_NO_SPACESHIP_OPERATOR
)

target_compile_definitions(${PROJECT_NAME} PUBLIC ${VULKAN_HPP_DEFINITIONS})

# Includes
set(INCLUDE_DIRS
	External/Vma/include
)

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

# Sources
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "Source/*.cpp")

set(ALL_SOURCES
	${SOURCES}
)

target_sources(${PROJECT_NAME} PUBLIC ${ALL_SOURCES})

# Modules
file(GLOB_RECURSE MODULE_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "Source/*.cppm")

target_sources(${PROJECT_NAME} PUBLIC
	FILE_SET CXX_MODULES
	BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${MODULE_SOURCES}
)

# Optional
target_compile_features(
	${PROJECT_NAME}
	PRIVATE   cxx_std_23
	INTERFACE cxx_std_20
)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# set(SOME_OPTION ON)
# add_subdirectory()

# target_link_libraries(${PROJECT_NAME} PRIVATE
# 	SOME_LIBRARY::PARTITION
# )

if (NG_BUILD_SAMPLES)
	add_subdirectory(Samples)
endif()

# cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-stdlib=libc++ -fno-rtti -fno-exceptions" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNG_BUILD_SAMPLES=ON
# cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
# cmake --build build