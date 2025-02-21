set(EXAMPLE_NAME HostNetwork)

file(GLOB_RECURSE TEST_SOURCE_FILES "*.cpp")

add_executable(${EXAMPLE_NAME} ${TEST_SOURCE_FILES})

target_link_libraries(${EXAMPLE_NAME} NeuralGraphics::NeuralGraphics)