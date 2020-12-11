file(GLOB TESTS_CPP *.cpp)
foreach(testsourcefile ${TESTS_CPP})

    get_filename_component(testname ${testsourcefile} NAME)
    string(REPLACE ".cpp" "" testname ${testname})

    #Add executable
    add_executable(${testname} ${testsourcefile})

    target_link_libraries(${testname} ${Trilinos_LIBRARIES})
    target_link_libraries(${testname} ${VTK_LIBRARIES})
    target_link_libraries(${testname} ${ias_LIBRARIES})
    target_link_libraries(${testname} ${OpenMP_LIBRARIES})

    #    install(TARGETS ${testName} DESTINATION ${ias_BIN_INSTALL})
    set_target_properties(${testname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ias_BIN_INSTALL})
endforeach(testsourcefile ${TESTS_CPP})
