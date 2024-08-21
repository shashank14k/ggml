#
# simple-ctx
if (GGML_INCLUDE_DIR)
    include_directories(${GGML_INCLUDE_DIR})
endif()

if (GGML_LIB_DIR)
    link_directories(${GGML_LIB_DIR})
endif()

set(TEST_TARGET simple-ctx-cpp)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

set(TEST_TARGET simple-ctx-c)
add_executable(${TEST_TARGET} simple-ctx.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

#
# simple-backend

#set(TEST_TARGET simple-backend)
#add_executable(${TEST_TARGET} simple-backend.cpp)
#target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
