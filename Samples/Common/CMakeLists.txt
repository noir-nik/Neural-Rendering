set(SAMPLE_NAME SamplesCommon)

file(GLOB_RECURSE TEST_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${SAMPLE_NAME} ${TEST_SOURCE_FILES})

target_link_libraries(${SAMPLE_NAME} PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

set(INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

target_include_directories(${SAMPLE_NAME} PUBLIC ${INCLUDE_DIRS})

# Modules
file(GLOB_RECURSE MODULE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cppm")

target_sources(${SAMPLE_NAME} PUBLIC
	FILE_SET CXX_MODULES
	BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${MODULE_SOURCES}
)

if(NOT TARGET glfw)
add_subdirectory(${GLFW_DIR} ${PROJECT_SOURCE_DIR}/External/Glfw ${CMAKE_CURRENT_BINARY_DIR}/External/Glfw)
target_link_libraries(${SAMPLE_NAME} PRIVATE glfw)
endif()

function(generate_spv_shader_glsl input_file output_file)
	add_custom_command(
		OUTPUT ${output_file}
		COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${input_file} -fentry-point=main --target-spv=spv1.6 -o ${output_file} -g
		DEPENDS ${input_file}
		COMMENT "Compiling SPIR-V: ${input_file}"
		VERBATIM
		COMMAND_EXPAND_LISTS
	)
endfunction()

function(generate_spv_shader_slang input_file output_file)
	add_custom_command(
		OUTPUT ${output_file}
		COMMAND slangc ${input_file} -entry main -target spirv -o ${output_file} -depfile ${output_file}.d -I${CMAKE_CURRENT_SOURCE_DIR}/../../Source/Shaders
		DEPENDS ${input_file}
		COMMENT "Compiling SPIR-V: ${input_file}"
		DEPFILE ${output_file}.d
		VERBATIM
		COMMAND_EXPAND_LISTS
	)
endfunction()

set(COMMON_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders CACHE PATH "Path to common shaders")
set(COMMON_SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Shaders CACHE PATH "Path to common shaders output")

generate_spv_shader_glsl(${COMMON_SHADERS_DIR}/Quad.vert ${COMMON_SHADERS_OUTPUT_DIR}/Quad.vert.spv)

add_custom_target(common_shaders_spv ALL DEPENDS
	${COMMON_SHADERS_OUTPUT_DIR}/Quad.vert.spv
)
add_dependencies(${SAMPLE_NAME} shaders_spv)
