####################
#      Global      #
####################

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")


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
    graphics/camera.h
    graphics/raytracing/ray-generation.cuh
    graphics/raytracing/ray-tracing.cuh
    math/matrix_math.cuh
    math/vector_math.cuh
    math/rt_math.cuh
    util/render-data-displayer.h
    util/grid-layout.h
    util/common.cuh
    util/buffer.cuh
    util/log.h
    voxel/material.h
    voxel/chunk.h
)

# Source files (relative to "src" directory)
set(SOURCES
    application/application.cpp
    graphics/window.cpp
    graphics/camera.cpp
    util/log.cpp
    util/render-data-displayer.cpp
    util/grid-layout.cpp
    util/buffer.cu
    main.cu
)

# Compiler definitions
set(DEFINES
    -DGLEW_STATIC
    -DCMAKE_CUDA_FLAGS="-arch=sm_60"
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
