message(STATUS "[bSpMV] Scanning matrix format directory")

set(SOURCES 
        Matrix.cpp
        MatrixBase.cpp 
        MatrixCOO.cpp
        MatrixCSR.cpp 
        MatrixELL.cpp
        MatrixHLL.cpp
        MatrixBwcCoo.cpp
        MatrixBwcCsr.cpp 
        MatrixBwcEll.cpp
        MatrixBwcHll.cpp
    )

add_library(format ${SOURCES})
target_include_directories(format PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(format PUBLIC)
