file(GLOB TEST_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
list(FILTER TEST_DIRS EXCLUDE REGEX "CMakeLists.txt")
foreach(example_dir ${TEST_DIRS})
    add_subdirectory(${example_dir})
endforeach()