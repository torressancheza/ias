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
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/Programs/tissueMech" CACHE PATH "Install Directory" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(MPI_BASE_PATH "$ENV{HOME}/local/openmpi" CACHE PATH "MPI base directory")
set(TRI_BASE_PATH "$ENV{HOME}/local/trilinos" CACHE PATH "Trilinos base directory")
set(VTK_BASE_PATH "$ENV{HOME}/local/vtk" CACHE PATH "Vtk base directory")

#MPI
#Library
find_library(MPI_LIBRARIES NAMES mpi HINTS ${MPI_BASE_PATH}/lib ${MPI_BASE_PATH}/lib64)
include_directories(SYSTEM ${MPI_BASE_PATH}/include)

#Compuler flags
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_FLAGS "-std=c++11 -Xpreprocessor -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -ggdb -O0 -Wall -Wextra -Wpedantic")

#Trilinos
set(Trilinos_DIR ${TRI_BASE_PATH}/lib/cmake/Trilinos/)
set(CMAKE_PREFIX_PATH ${Trilinos_DIR} ${CMAKE_PREFIX_PATH})
find_package(Trilinos REQUIRED)
include_directories(SYSTEM ${TRI_BASE_PATH}/include)

#VTK
file(GLOB VTK_DIR ${VTK_BASE_PATH}/lib/cmake/vtk-*/)
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

add_subdirectory(${ias_SOURCE_DIR}/Extern)
add_subdirectory(${ias_SOURCE_DIR}/Tensor)
add_subdirectory(${ias_SOURCE_DIR}/BasisFunctions)
add_subdirectory(${ias_SOURCE_DIR}/Cell)
add_subdirectory(${ias_SOURCE_DIR}/Tissue)
add_subdirectory(${ias_SOURCE_DIR}/Integration)
add_subdirectory(${ias_SOURCE_DIR}/Solvers)
add_subdirectory(${ias_SOURCE_DIR}/Tests)
add_subdirectory(${ias_SOURCE_DIR}/Tutorials)
