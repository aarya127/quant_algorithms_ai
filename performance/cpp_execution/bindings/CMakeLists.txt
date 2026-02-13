pybind11_add_module(cpp_bindings bindings.cpp)

target_link_libraries(cpp_bindings PRIVATE
    execution_engine
    order_book
    common
)

# Set output directory
set_target_properties(cpp_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../
)
