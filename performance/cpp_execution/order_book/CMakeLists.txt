add_library(order_book STATIC
    order_book.cpp
)

target_include_directories(order_book PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(order_book
    common
)
