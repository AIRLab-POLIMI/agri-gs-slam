cmake_minimum_required(VERSION 3.12)
project(scancontext)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(include)
include_directories(${PCL_INCLUDE_DIRS})

# Source files
set(SCANCONTEXT_SOURCES
    src/scancontext/scancontext.cpp
)

# Create static library
add_library(scancontext_lib STATIC
    ${SCANCONTEXT_SOURCES}
)

target_link_libraries(scancontext_lib
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

target_compile_definitions(scancontext_lib PRIVATE ${PCL_DEFINITIONS})

# Python bindings
pybind11_add_module(scancontext bindings/pybind.cpp)

target_link_libraries(scancontext PRIVATE
    scancontext_lib
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

target_compile_definitions(scancontext PRIVATE ${PCL_DEFINITIONS})

# Set module name to "scancontext" (no prefix)
set_target_properties(scancontext PROPERTIES
    OUTPUT_NAME "scancontext"
    PREFIX ""
)

# Get the correct Python path
execute_process(
    COMMAND python3 -c "import sys; print([p for p in sys.path if 'local' in p and 'packages' in p][0])"
    OUTPUT_VARIABLE PYTHON_LOCAL_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS scancontext
    LIBRARY DESTINATION ${PYTHON_LOCAL_PACKAGES}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
