include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB_RECURSE ALL_EXE_SRCS 
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/**/*.c
)

foreach(EXE_PATH ${ALL_EXE_SRCS})
  get_filename_component(EXE_NAME ${EXE_PATH} NAME_WE)
  get_filename_component(EXT_NAME ${EXE_PATH} EXT)
  if (EXT_NAME STREQUAL ".c")
    set(EXE_NAME "${EXE_NAME}_c")
  endif()
  add_executable(${EXE_NAME} ${EXE_PATH})
  target_link_libraries(${EXE_NAME} ${PROJECT_NAME})
endforeach()