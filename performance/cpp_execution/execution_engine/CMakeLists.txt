add_library(execution_engine STATIC
    execution_engine.cpp
)

target_include_directories(execution_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(execution_engine
    common
)
