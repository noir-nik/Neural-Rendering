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


set(NLOHMANN_JSON_BUILD_MODULES ON)
add_subdirectory(${NLOHMANN_JSON_DIR} ${PROJECT_SOURCE_DIR}/External/nlohmann_json ${CMAKE_CURRENT_BINARY_DIR}/External/json)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json_modules)

add_subdirectory(${TIN_DIR} ${PROJECT_SOURCE_DIR}/External/tinyobjloader  ${CMAKE_CURRENT_BINARY_DIR}/External/obj)
target_link_libraries(${PROJECT_NAME} PUBLIC tinyobjloader )

set(INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/HeaderNames
	${CMAKE_CURRENT_SOURCE_DIR}/HeaderNames/FASTKAN
	${CMAKE_CURRENT_SOURCE_DIR}/HeaderNames/NBRDF
)

target_include_directories(${SAMPLE_NAME} PUBLIC ${INCLUDE_DIRS})

target_link_libraries(${SAMPLE_NAME} PRIVATE SamplesCommon)

set(${SAMPLE_NAME}_SHADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Shaders CACHE PATH "${SAMPLE_NAME} shaders")
set(${SAMPLE_NAME}_SHADERS_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/Shaders CACHE PATH "${SAMPLE_NAME} shaders output")

set(INCLUDE_SLANG
	${${PROJECT_NAME}_SHADERS_DIR}
	${${SAMPLE_NAME}_SHADERS_DIR}
	${${SAMPLE_NAME}_SHADERS_DIR}/MLP
	${${SAMPLE_NAME}_SHADERS_DIR}/KAN
	${${SAMPLE_NAME}_SHADERS_DIR}/KAN/CoopVec
	${${SAMPLE_NAME}_SHADERS_DIR}/KAN/Headers
	${${SAMPLE_NAME}_SHADERS_DIR}/KAN/Headers/FASTKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/FASTKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/FASTKAN/2604/v2
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/NEWFASTKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/NBRDF
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/SINEKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/CHEBYKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/RELUKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/CoopVec
)

set(OPTIONS_SLANG -Wno-30081 -Wno-39001 -O3)

set(SLANG_DEPENDS
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain.slang.spv
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Skybox.slang.spv
)

generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/BRDFMain.slang ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain.slang.spv
	INCLUDE ${INCLUDE_SLANG}
	OPTIONS ${OPTIONS_SLANG}
)

generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/Skybox.slang ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Skybox.slang.spv
	INCLUDE ${INCLUDE_SLANG}
	OPTIONS ${OPTIONS_SLANG}
)

# MODELS_PATH

option(MODELS_PATH "KAN header files" OFF)

if(NOT MODELS_PATH)
    # message(FATAL_ERROR "MODELS_PATH is not set")
	set(GLOB MODELS_FILES_SLANG ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/KAN/Headers/BRDFFASTKAN.h)
else()
	file(GLOB MODELS_FILES_SLANG "${MODELS_PATH}/*.slang")
endif()



set(${SAMPLE_NAME}_SHADERS_GENERATED_DIR ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Generated)

# message("{{SAMPLE_NAME}_SHADERS_GENERATED_DIR}" ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR})
# Add C++ MODELS_PATH
target_compile_definitions(${SAMPLE_NAME} PUBLIC GENERATED_DIR="${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}")

foreach(model ${MODELS_FILES_SLANG})
	# message("model:" ${model})

	# Get basename
	get_filename_component(model_basename_we "${model}" NAME_WE)
	set(OPTIONS_SLANG_model ${OPTIONS_SLANG})

	# Get num layers
	string(REGEX MATCHALL "[0-9]+" all_numbers "${model_basename_we}")
	list(GET all_numbers -3 NUM_LAYERS)
	
	# Add options for weights header
	list(APPEND OPTIONS_SLANG_model -DINC="${model_basename_we}" "-I${MODELS_PATH}" -DLAYERS=3)

	# Output name
	set(OUTPATH ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}/${model_basename_we}.slang.spv)
	
    generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/BRDFMain.slang ${OUTPATH}
		INCLUDE ${INCLUDE_SLANG}
		OPTIONS ${OPTIONS_SLANG_model}
	)
	list(APPEND SLANG_DEPENDS ${OUTPATH})
endforeach()



add_custom_target(${SAMPLE_NAME}_shaders_spv ALL DEPENDS ${SLANG_DEPENDS})

add_dependencies(${SAMPLE_NAME} ${SAMPLE_NAME}_shaders_spv)



# Copy assets

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


