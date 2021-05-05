#Initialise
project(ias)

cmake_minimum_required(VERSION 3.12)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(MAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

cmake_policy(SET CMP0025 NEW)

#Options and flags
option(BUILD_SHARED_LIBS "Build ias with shared libraries." ON)
option(CMAKE_BUILD_TYPE "Build Type: Debug or Release" Release)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/Programs/ias" CACHE PATH "Install Directory" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


#MPI
find_package(MPI REQUIRED)

#OpenMP
if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP REQUIRED)

#Compuler flags
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -ggdb -O0 -Wall -Wextra -Wpedantic")

#Trilinos
find_package(Trilinos REQUIRED)

#VTK
find_package(VTK REQUIRED)

#Set installation paths
set(ias_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})
set(ias_LIBS_INSTALL ${CMAKE_INSTALL_PREFIX}/lib)
set(ias_INCLUDE_INSTALL ${CMAKE_INSTALL_PREFIX}/include)
set(ias_BIN_INSTALL ${CMAKE_INSTALL_PREFIX}/bin)

#Set library
set(ias_LIBRARIES CACHE INTERNAL "ias_LIBRARIES")

#Set include directories
file(GLOB ias_INC_DIRS ${ias_SOURCE_DIR}/*)
include_directories(${ias_INC_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})

add_subdirectory(${ias_SOURCE_DIR}/Extern)
add_subdirectory(${ias_SOURCE_DIR}/Tensor)
add_subdirectory(${ias_SOURCE_DIR}/BasisFunctions)
add_subdirectory(${ias_SOURCE_DIR}/Cell)
add_subdirectory(${ias_SOURCE_DIR}/Tissue)
add_subdirectory(${ias_SOURCE_DIR}/Integration)
add_subdirectory(${ias_SOURCE_DIR}/Solvers)
add_subdirectory(${ias_SOURCE_DIR}/Auxiliary)
add_subdirectory(${ias_SOURCE_DIR}/Tests)
add_subdirectory(${ias_SOURCE_DIR}/Tutorials)
