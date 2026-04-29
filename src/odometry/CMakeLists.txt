cmake_minimum_required(VERSION 3.12)
project(odometry)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_LOCAL_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python site-packages path: ${PYTHON_LOCAL_PACKAGES}")

# Include directories
include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})

# Source files
set(DLO_SOURCES
    src/dlo/dlo.cpp
)

set(NANO_GICP_SOURCES
    src/nano_gicp/lsq_registration.cc
    src/nano_gicp/nano_gicp.cc
    src/nano_gicp/nanoflann.cc
)

# Create static library
add_library(odometry_lib STATIC
    ${DLO_SOURCES}
    ${NANO_GICP_SOURCES}
)

target_link_libraries(odometry_lib
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

target_compile_definitions(odometry_lib PRIVATE ${PCL_DEFINITIONS})

# Python bindings
pybind11_add_module(odometry bindings/pybind.cpp)

target_link_libraries(odometry PRIVATE
    odometry_lib
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

target_compile_definitions(odometry PRIVATE ${PCL_DEFINITIONS})

# Set module name to "odometry" (no prefix)
set_target_properties(odometry PROPERTIES
    OUTPUT_NAME "odometry"
    PREFIX ""
)

install(TARGETS odometry
    LIBRARY DESTINATION ${PYTHON_LOCAL_PACKAGES}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
