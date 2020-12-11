file(GLOB SOURCES_H   *.h)
file(GLOB SOURCES_CPP *.cpp)

set(SOURCE_FILES ${SOURCES_H} ${SOURCES_CPP})
set(LIBNAME iasIntegration)


add_library(${LIBNAME} ${SOURCE_FILES})

target_link_libraries(${LIBNAME} MPI::MPI_CXX)
target_link_libraries(${LIBNAME} ${VTK_LIBRARIES})
target_link_libraries(${LIBNAME} ${Trilinos_LIBRARIES})
target_link_libraries(${LIBNAME} ${ias_LIBRARIES})
target_link_libraries(${LIBNAME} OpenMP::OpenMP_CXX)

install(TARGETS ${LIBNAME} DESTINATION ${ias_LIBS_INSTALL})
install(FILES ${SOURCES_H} DESTINATION ${ias_INCLUDE_INSTALL})

set(ias_LIBRARIES ${ias_LIBRARIES} ${LIBNAME} CACHE INTERNAL "ias_LIBRARIES")
