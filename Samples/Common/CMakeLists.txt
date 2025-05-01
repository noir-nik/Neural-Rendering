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
	set(options "")
	set(oneValueArgs "ENTRY")
	set(multiValueArgs "INCLUDE;OPTIONS")
	cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

	set(include_string "")
	foreach(include_path IN LISTS ARG_INCLUDE)
		list(APPEND include_string -I${include_path})
	endforeach()

	set(command_options "")

	if (ARG_ENTRY)
		list(APPEND command_options -entry ${ARG_ENTRY})
	endif()
	if (include_string)
		list(APPEND command_options ${include_string})
	endif()
	if (ARG_OPTIONS)
		list(APPEND command_options ${ARG_OPTIONS})
	endif()

	message(STATUS "Compile command: ${command_options}")

	set(SLANGC_EXECUTABLE slangc)

	add_custom_command(
		OUTPUT ${output_file}
		COMMAND ${SLANGC_EXECUTABLE} ${input_file} -target spirv -o ${output_file} -depfile ${output_file}.d  ${command_options}
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
add_dependencies(${SAMPLE_NAME} common_shaders_spv)
