set(SAMPLE_NAME BRDF)

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

set(${SAMPLE_NAME}_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders CACHE PATH "${SAMPLE_NAME} shaders")
set(${SAMPLE_NAME}_SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Shaders CACHE PATH "${SAMPLE_NAME} shaders output")

generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/BRDFMain.slang ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain.slang.spv
	INCLUDE ${${PROJECT_NAME}_SHADERS_DIR} ${${SAMPLE_NAME}_SHADERS_DIR} ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}
	OPTIONS "-Wno-30081" "-Wno-39001"
)
# -Wno-30081 

add_custom_target(${SAMPLE_NAME}_shaders_spv ALL DEPENDS
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain.slang.spv
)

add_dependencies(${SAMPLE_NAME} ${SAMPLE_NAME}_shaders_spv)


set(WEIGHTS_FILE Assets/simple_brdf_weights.bin)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Assets/simple_brdf_weights.bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${WEIGHTS_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHTS_FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${WEIGHTS_FILE}
    VERBATIM
)

add_custom_target(${SAMPLE_NAME}_copy_assets ALL DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHTS_FILE}
)

add_dependencies(${SAMPLE_NAME} ${SAMPLE_NAME}_copy_assets)
