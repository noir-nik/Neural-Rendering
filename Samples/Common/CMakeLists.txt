set(SAMPLE_NAME SamplesCommon)

file(GLOB_RECURSE TEST_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${SAMPLE_NAME} ${TEST_SOURCE_FILES})

target_link_libraries(${SAMPLE_NAME} PRIVATE NeuralGraphics::NeuralGraphics)

set(GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../External/Glfw)

set(INCLUDE_DIRS
	${GLFW_DIR}/include
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
add_subdirectory(${GLFW_DIR} ../External/Glfw)
target_link_libraries(${SAMPLE_NAME} PRIVATE glfw)
endif()
