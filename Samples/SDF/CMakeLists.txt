set(SAMPLE_NAME SDF)

file(GLOB_RECURSE SAMPLE_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_executable(${SAMPLE_NAME} ${SAMPLE_SOURCE_FILES})

target_link_libraries(${SAMPLE_NAME} PRIVATE NeuralGraphics::NeuralGraphics)

set(INCLUDE_DIRS
)

target_include_directories(${SAMPLE_NAME} PUBLIC ${INCLUDE_DIRS})

target_link_libraries(${SAMPLE_NAME} PRIVATE SamplesCommon)

function(generate_spv_shader input_file output_file)
	add_custom_command(
		OUTPUT ${output_file}
		COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${input_file} -fentry-point=main --target-spv=spv1.6 -o ${output_file} -g
		DEPENDS ${input_file}
		COMMENT "Compiling SPIR-V: ${input_file}"
		VERBATIM
		COMMAND_EXPAND_LISTS
	)
endfunction()

set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders)
set(SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Shaders)

generate_spv_shader(${SHADERS_DIR}/Quad.vert ${SHADERS_OUTPUT_DIR}/Quad.vert.spv)
generate_spv_shader(${SHADERS_DIR}/SimpleSdf.vert ${SHADERS_OUTPUT_DIR}/SimpleSdf.vert.spv)
generate_spv_shader(${SHADERS_DIR}/SimpleSdf.frag ${SHADERS_OUTPUT_DIR}/SimpleSdf.frag.spv)
generate_spv_shader(${SHADERS_DIR}/SimpleSdfDebug.frag ${SHADERS_OUTPUT_DIR}/SimpleSdfDebug.frag.spv)

add_custom_target(shaders_spv ALL DEPENDS
	${SHADERS_OUTPUT_DIR}/Quad.vert.spv
	${SHADERS_OUTPUT_DIR}/SimpleSdf.vert.spv
	${SHADERS_OUTPUT_DIR}/SimpleSdf.frag.spv
	${SHADERS_OUTPUT_DIR}/SimpleSdfDebug.frag.spv
)
add_dependencies(${SAMPLE_NAME} shaders_spv)
