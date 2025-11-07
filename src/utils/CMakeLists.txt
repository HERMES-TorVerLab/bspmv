message(STATUS "[bSpMV] Scanning utils directory")

set(SOURCES 
        MatrixCheck.cpp
    )

add_library(bspmvUtils ${SOURCES})
target_include_directories(bspmvUtils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bspmvUtils PUBLIC)
