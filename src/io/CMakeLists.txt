message(STATUS "[bSpMV] Scanning matrix io directory")

set(SOURCES 
        MatrixReader.cpp
        MatrixConverter.cpp
    )


add_library(io ${SOURCES})
target_include_directories(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(io PUBLIC format)
