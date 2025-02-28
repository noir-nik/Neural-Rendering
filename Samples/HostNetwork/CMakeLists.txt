set(SAMPLE_NAME HostNetwork)

file(GLOB_RECURSE SAMPLE_SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_executable(${SAMPLE_NAME} ${SAMPLE_SOURCE_FILES})

target_link_libraries(${SAMPLE_NAME} PRIVATE NeuralGraphics::NeuralGraphics)
