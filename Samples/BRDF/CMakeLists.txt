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

set(${SAMPLE_NAME}_SHADERS_GENERATED_DIR ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Generated)
file(RELATIVE_PATH ${SAMPLE_NAME}_SHADERS_GENERATED_DIR_RELATIVE "${CMAKE_CURRENT_BINARY_DIR}" ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR})   
# BRDFModels.def
set(${SAMPLE_NAME}_SHADERS_GENERATED_MODELS_DEF ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}/BRDFModels.def)

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
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/NEWFASTKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/NBRDF
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/SINEKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/CHEBYKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/RELUKAN
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/CoopVec
)

set(OPTIONS_SLANG -Wno-30081 -Wno-39001 -O3)

set(SLANG_DEPENDS
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain-point.slang.spv
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain-env.slang.spv
	${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Skybox.slang.spv
)

generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/BRDFMain.slang ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain-point.slang.spv
	INCLUDE ${INCLUDE_SLANG}
	OPTIONS ${OPTIONS_SLANG} -DIS_POINT=1
)

generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/BRDFMain.slang ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/BRDFMain-env.slang.spv
	INCLUDE ${INCLUDE_SLANG}
	OPTIONS ${OPTIONS_SLANG} -DIS_POINT=0
)

generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/Skybox.slang ${${SAMPLE_NAME}_SHADERS_OUTPUT_DIR}/Skybox.slang.spv
	INCLUDE ${INCLUDE_SLANG}
	OPTIONS ${OPTIONS_SLANG}
)

# Function write_def_file
function(write_def_file in_brdf_models filename)
	set(brdf_models ${${in_brdf_models}})

	foreach(fname ${brdf_models})
		list(APPEND br_names "BRDF_NAME(${fname})")
	endforeach()

	# Convert list to newline-separated string
	string(JOIN "\n" list_as_string_brdf ${br_names})

	# set out
	set(outstr
"\
#ifndef BRDF_NAME
#define BRDF_NAME(x)
#endif

${list_as_string_brdf}

"
# undef BRDF_NAME
)
	# Write file
	file(WRITE ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}/${filename}
		${outstr}
	)
	message("Wrote " ${filename})
endfunction()

# MODELS_PATH

option(MODELS_PATH "KAN header files" OFF)


if(MODELS_PATH)
	file(GLOB MODELS_FILES_SLANG "${MODELS_PATH}/*.slang")
else()
	message(WARNING "MODELS_PATH is not set")
	set(MODELS_FILES_SLANG ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/KAN/Headers/BRDFFASTKAN_2_5_nCps8.slang)
endif()

# message(MODELS_PATH, ${MODELS_PATH})
# message(MODELS_FILES_SLANG, ${MODELS_FILES_SLANG})




list(LENGTH MODELS_FILES_SLANG MODELS_FILES_SLANG_length)
if(MODELS_FILES_SLANG_length EQUAL 0)
	message(FATAL_ERROR "No .slang models found in ${MODELS_PATH}")
endif()

list(SORT MODELS_FILES_SLANG COMPARE NATURAL ORDER ASCENDING)

foreach(model ${MODELS_FILES_SLANG})

	# Get basename
	get_filename_component(model_basename_we "${model}" NAME_WE)
	set(model_basename_we_point ${model_basename_we}-point)
	set(model_basename_we_env ${model_basename_we}-env)

	# Get num layers
	# string(REGEX MATCHALL "[0-9]+" all_numbers "${model_basename_we}")
	# list(GET all_numbers -3 NUM_LAYERS)
	# math(EXPR NUM_LAYERS "${NUM_LAYERS} + 1")
	# message(${model_basename_we})
	set(NUM_LAYERS 3)
	
	# Add options for weights header
	set(OPTIONS_SLANG_model ${OPTIONS_SLANG})
	list(APPEND OPTIONS_SLANG_model -DINC="${model_basename_we}" "-I${MODELS_PATH}" -DLAYERS=${NUM_LAYERS})
	if("${model_basename_we}" MATCHES "NBRDF")
		message(STATUS "String contains substring")
		list(APPEND OPTIONS_SLANG_model -DNBRDF -DFF=${model_basename_we})
	endif() 

	# Output name
	set(OUTPATH_POINT ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}/${model_basename_we_point}.slang.spv)
	set(OUTPATH_ENV   ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}/${model_basename_we_env}.slang.spv)
	
	# Generate
    generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/BRDFMain.slang ${OUTPATH_POINT}
		INCLUDE ${INCLUDE_SLANG}
		OPTIONS ${OPTIONS_SLANG_model} -DIS_POINT=1
	)

	generate_spv_shader_slang(${${SAMPLE_NAME}_SHADERS_DIR}/BRDFMain.slang ${OUTPATH_ENV}
		INCLUDE ${INCLUDE_SLANG}
		OPTIONS ${OPTIONS_SLANG_model} -DIS_POINT=0
	)

	# Append lists
	list(APPEND SLANG_DEPENDS ${OUTPATH_POINT} ${OUTPATH_ENV})
	list(APPEND BRDF_MODEL_NAMES_POINT "${model_basename_we_point}")
	list(APPEND BRDF_MODEL_NAMES_ENV "${model_basename_we_env}")
	list(APPEND BRDF_MODEL_NAMES "${model_basename_we}")
endforeach()
	
# message(BRDF_MODEL_NAMES_POINT "${BRDF_MODEL_NAMES_POINT}")
# message(BRDF_MODEL_NAMES_ENV "${BRDF_MODEL_NAMES_ENV}")
# message(BRDF_MODEL_NAMES "${BRDF_MODEL_NAMES}")

# list(TRANSFORM my_list REPLACE "\\.txt$" ".dat")

write_def_file(BRDF_MODEL_NAMES_POINT BRDFModels-point.def)
write_def_file(BRDF_MODEL_NAMES_ENV BRDFModels-env.def)

file(WRITE ${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}/BRDFModels.def
"\
#include \"BRDFModels-point.def\"
#include \"BRDFModels-env.def\"
"
)

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

# Add C++ MODELS_PATH
target_compile_definitions(${SAMPLE_NAME} PUBLIC
	GENERATED_DIR="${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}"
	GENERATED_DIR_RELATIVE="${${SAMPLE_NAME}_SHADERS_GENERATED_DIR_RELATIVE}"
	GENERATED_MODELS_DEF="${${SAMPLE_NAME}_SHADERS_GENERATED_MODELS_DEF}"
	GENERATED_MODELS_COUNT=${MODELS_FILES_SLANG_length}
)

target_include_directories(${SAMPLE_NAME} PUBLIC
	${${SAMPLE_NAME}_SHADERS_GENERATED_DIR}
)