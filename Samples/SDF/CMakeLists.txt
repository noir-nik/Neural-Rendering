set(SAMPLE_NAME SDF)

file(GLOB_RECURSE SAMPLE_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_executable(${SAMPLE_NAME} ${SAMPLE_SOURCE_FILES})

file(GLOB_RECURSE MODULE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cppm")

target_sources(${SAMPLE_NAME} PUBLIC
	FILE_SET CXX_MODULES
	BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${MODULE_SOURCES}
)

target_link_libraries(${SAMPLE_NAME} PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

set(INCLUDE_DIRS
)

target_include_directories(${SAMPLE_NAME} PUBLIC ${INCLUDE_DIRS})

target_link_libraries(${SAMPLE_NAME} PRIVATE SamplesCommon)

set(SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders CACHE PATH "Path to shaders")
set(SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Shaders CACHE PATH "Path to shaders output")

generate_spv_shader_slang(${SHADERS_DIR}/SdfMain.slang ${SHADERS_OUTPUT_DIR}/SdfMain.slang.spv)

add_custom_target(shaders_spv ALL DEPENDS
	${SHADERS_OUTPUT_DIR}/Quad.vert.spv
	${SHADERS_OUTPUT_DIR}/SdfMain.slang.spv
)

add_dependencies(${SAMPLE_NAME} shaders_spv)

add_custom_command(
	OUTPUT ${SHADERS_OUTPUT_DIR}/Quad.vert.spv
	COMMAND ${CMAKE_COMMAND} -E copy ${COMMON_SHADERS_OUTPUT_DIR}/Quad.vert.spv ${SHADERS_OUTPUT_DIR}/Quad.vert.spv
	DEPENDS ${COMMON_SHADERS_OUTPUT_DIR}/Quad.vert.spv
	VERBATIM
)
