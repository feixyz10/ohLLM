cmake_minimum_required(VERSION 3.15)
project(ohllm VERSION 0.1)

####### OPTIONS
option(BUILD_DEBUG "build debug" OFF)
option(BUILD_LOG "build log" OFF)

message(STATUS "BUILD_DEBUG: ${BUILD_DEBUG}")
message(STATUS "BUILD_LOG: ${BUILD_LOG}")
if (BUILD_LOG)
  add_definitions("-DBUILD_LOG")
endif()

if (BUILD_DEBUG) 
  set(CMAKE_BUILD_TYPE "Debug")
  add_compile_options(-g -Wall -Wextra -Wno-gnu-statement-expression)
else()
  set(CMAKE_BUILD_TYPE "Release")
  add_compile_options(-O3 -Wall -Wextra -Wno-gnu-statement-expression)
endif()
set(CMAKE_CXX_STANDARD 20)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB_RECURSE ALL_SRC_FILES 
  ${CMAKE_CURRENT_SOURCE_DIR}/ohllm/**/*.cc
)
add_library(${PROJECT_NAME} STATIC 
  ${ALL_SRC_FILES}
)

function(add_subdirectory_if_ok dir)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    message(STATUS "Build ${dir}")
    add_subdirectory(${dir})
  else()
    message(STATUS "Skip ${dir}")
  endif()
endfunction(add_subdirectory_if_ok)

add_subdirectory_if_ok(test)
