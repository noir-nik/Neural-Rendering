cmake_minimum_required(VERSION 3.30)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")

project(NeuralRendering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_MODULE_STD 1)

option(NG_BUILD_SAMPLES "Build samples" ON)

# Sources
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "Source/*.cpp")

set(ALL_SOURCES
	${SOURCES}
)

add_library(${PROJECT_NAME} ${SOURCES})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

find_package(Vulkan REQUIRED)
if(${Vulkan_VERSION} VERSION_LESS "1.4.309" )
message( WARNING "Minimum required Vulkan version is 1.4.309. Found ${Vulkan_VERSION}.")
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
	Source
)

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})

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

# Add Vulkan module
add_library( VulkanHppModule )
target_sources( VulkanHppModule PUBLIC
	FILE_SET CXX_MODULES
	BASE_DIRS ${Vulkan_INCLUDE_DIR}
	FILES ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
)
target_compile_features( VulkanHppModule PUBLIC cxx_std_20 )
target_link_libraries( VulkanHppModule PUBLIC Vulkan::Vulkan )

target_link_libraries( ${PROJECT_NAME} PUBLIC VulkanHppModule )
target_compile_definitions(VulkanHppModule PUBLIC ${VULKAN_HPP_DEFINITIONS})

# Sanitize
option(NG_USE_SANITIZER "Use sanitizer" OFF)
if (NG_USE_SANITIZER)
	set(NG_SANITIZER_FLAGS "-fsanitize=address -fsanitize=undefined -fsanitize=leak")
	if (MSVC)
		set(NG_SANITIZER_FLAGS "/fsanitize=address")
	endif()
	add_compile_options(${NG_SANITIZER_FLAGS})
	add_link_options(${NG_SANITIZER_FLAGS})
endif()

set(${PROJECT_NAME}_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Source/Shaders CACHE PATH "${PROJECT_NAME} shaders source")
set(${PROJECT_NAME}_SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Source/Shaders CACHE PATH "${PROJECT_NAME} shaders output")

# Samples
if (NG_BUILD_SAMPLES)
	add_subdirectory(Samples)
endif()

# cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-stdlib=libc++ -fno-rtti -fno-exceptions" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNG_BUILD_SAMPLES=ON
# cmake . -B build/cl -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DNG_BUILD_SAMPLES=ON -DCMAKE_CXX_FLAGS="/D_HAS_EXCEPTIONS=0 /GR-"
# cmake --build build