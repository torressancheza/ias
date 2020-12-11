get_filename_component(folder ${CMAKE_CURRENT_LIST_DIR} NAME)

file(GLOB SOURCES_CPP *.cpp)
file(GLOB SOURCES_H *.h)
file(GLOB SOURCES ${SOURCES_CPP} ${SOURCES_H})

set(exName "ias${folder}")
add_executable(${exName} ${SOURCES})
target_link_libraries(${exName} MPI::MPI_CXX)
target_link_libraries(${exName} OpenMP::OpenMP_CXX)
target_link_libraries(${exName} ${Trilinos_LIBRARIES})
target_link_libraries(${exName} ${VTK_LIBRARIES})
target_link_libraries(${exName} ${ias_LIBRARIES})

install(TARGETS ${exName} DESTINATION ${ias_BIN_INSTALL})
