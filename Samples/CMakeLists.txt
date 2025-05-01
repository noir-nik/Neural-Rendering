foreach(dir Common SDF )
# foreach(dir Common SDF BRDF)
    add_subdirectory(${dir})
endforeach()