####################
#      Global      #
####################

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)



#####################
#      Project      #
#####################

# Project variables
set(LOCAL_PROJECT_NAME        "cuda-voxel-raytracing")
set(LOCAL_PROJECT_VERSION     "0.0.1")
set(LOCAL_PROJECT_DESCRIPTION "Description")

# Header files (relative to "include" directory)
set(HEADERS
    cuda-voxel-raytracing.h
    application/application.h
    graphics/window.h
    util/render-data-displayer.h
    util/buffer.cuh
    util/log.h
)

# Source files (relative to "src" directory)
set(SOURCES
    application/application.cpp
    graphics/window.cpp
    util/log.cpp
    util/render-data-displayer.cpp
    util/buffer.cu
    main.cu
)

# Compiler definitions
set(DEFINES
    -DGLEW_STATIC
)

# Compiler options
set(OPTIONS

)

# Project setup
project(${LOCAL_PROJECT_NAME}
        VERSION ${LOCAL_PROJECT_VERSION}
        DESCRIPTION ${LOCAL_PROJECT_DESCRIPTION}
        LANGUAGES CXX CUDA)

add_executable(${LOCAL_PROJECT_NAME})

list(TRANSFORM HEADERS PREPEND "include/")
list(TRANSFORM SOURCES PREPEND "src/")

target_include_directories(${LOCAL_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(${LOCAL_PROJECT_NAME} PRIVATE ${SOURCES} ${HEADERS})
target_compile_definitions(${LOCAL_PROJECT_NAME} PRIVATE ${DEFINES})
target_compile_options(${LOCAL_PROJECT_NAME} PRIVATE ${OPTIONS})

set_target_properties(${LOCAL_PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")



####################
#   Dependencies   #
####################

#find_package(OpenGL REQUIRED)

target_include_directories(${LOCAL_PROJECT_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	"deps/glfw/include"
	"deps/glew/include"
	"deps/glm"
)

target_link_libraries(${LOCAL_PROJECT_NAME} PUBLIC
    ${OPENGL_LIBRARY}
    opengl32.lib
	"${PROJECT_SOURCE_DIR}/deps/glfw/lib-vc2019/glfw3.lib"
	"${PROJECT_SOURCE_DIR}/deps/glew/lib/Release/x64/glew32s.lib"
)
