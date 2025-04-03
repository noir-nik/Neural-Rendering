set(SAMPLE_NAME SDF)

file(GLOB_RECURSE SAMPLE_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_executable(${SAMPLE_NAME} ${SAMPLE_SOURCE_FILES})

file(GLOB_RECURSE MODULE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cppm")

target_sources(${SAMPLE_NAME} PUBLIC
	FILE_SET CXX_MODULES
	BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${MODULE_SOURCES}
)

target_link_libraries(${SAMPLE_NAME} PRIVATE NeuralGraphics::NeuralGraphics)

set(INCLUDE_DIRS
)

target_include_directories(${SAMPLE_NAME} PUBLIC ${INCLUDE_DIRS})

target_link_libraries(${SAMPLE_NAME} PRIVATE SamplesCommon)


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

set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders)
set(SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Shaders)

generate_spv_shader_glsl(${SHADERS_DIR}/Quad.vert ${SHADERS_OUTPUT_DIR}/Quad.vert.spv)
generate_spv_shader_slang(${SHADERS_DIR}/SdfMain.slang ${SHADERS_OUTPUT_DIR}/SdfMain.slang.spv)

add_custom_target(shaders_spv ALL DEPENDS
	${SHADERS_OUTPUT_DIR}/Quad.vert.spv
	${SHADERS_OUTPUT_DIR}/SdfMain.slang.spv
)
add_dependencies(${SAMPLE_NAME} shaders_spv)
