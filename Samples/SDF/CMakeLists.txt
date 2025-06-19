set(SAMPLE_NAME SDF)

file(GLOB_RECURSE SAMPLE_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")


link_libraries(
	${PROJECT_NAME}::${PROJECT_NAME}
	SamplesCommon
)

# CUDNN sample
set(CUDA_SAMPLE_NAME cu${SAMPLE_NAME})
add_executable(${CUDA_SAMPLE_NAME} cuSDF.cpp)
if(WITH_CUDA)
	target_sources(${CUDA_SAMPLE_NAME} PRIVATE "SDF.cu")
	target_link_libraries(${CUDA_SAMPLE_NAME} PRIVATE ${PROJECT_NAME}-cuda)
endif()

# Vulkan sample
add_executable(${SAMPLE_NAME} "SDF.cpp") # ${SAMPLE_SOURCE_FILES}

file(GLOB_RECURSE MODULE_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cppm")

target_sources(${SAMPLE_NAME} PUBLIC
	FILE_SET CXX_MODULES
	BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
	FILES ${MODULE_SOURCES}
)

# Compile shaders
set(${SAMPLE_NAME}_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders CACHE PATH "${SAMPLE_NAME} shaders")
set(${SAMPLE_NAME}_SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Shaders CACHE PATH "${SAMPLE_NAME} shaders output")

generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/SdfMain.slang ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/SdfMain.slang.spv
	INCLUDE ${${PROJECT_NAME}_SHADERS_DIR}
)

add_custom_target(${SAMPLE_NAME}_shaders_spv ALL DEPENDS
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Quad.vert.spv
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/SdfMain.slang.spv
)

add_dependencies(${SAMPLE_NAME} ${SAMPLE_NAME}_shaders_spv)

add_custom_command(
	OUTPUT ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Quad.vert.spv
	COMMAND ${CMAKE_COMMAND} -E copy ${COMMON_SHADERS_OUTPUT_DIR}/Quad.vert.spv ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Quad.vert.spv
	DEPENDS ${COMMON_SHADERS_OUTPUT_DIR}/Quad.vert.spv
	VERBATIM
)
