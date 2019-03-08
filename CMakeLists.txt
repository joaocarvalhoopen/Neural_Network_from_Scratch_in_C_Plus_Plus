cmake_minimum_required(VERSION 3.0.0)
project(neural_network VERSION 0.1.0)

include(CTest)
enable_testing()


# Because by default CMAKE_BUILD_TYPE is not set, we set it.
# In VSCode this is'nt necessary.
if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE Release)
endif()

# Set the output directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../output)


# Says that we want an executable and the path to the main.cpp file.
# The following method after the comments is more generic.
#
# add_executable(neural_network ./src/main.cpp)

# Add all the files recursivelly that are bellow the directories and subdirectories of src and include.
file (GLOB_RECURSE SRC_FILES
	src/*.cpp
	include/*.h
)
add_executable (neural_network ${SRC_FILES})

# Log's to the console the files that were added. 
message ("Src files: " ${SRC_FILES})

# Set the compilation options passed to GCC. 
# In this case i want to use the C++17 Standard complience.
target_compile_options (neural_network PUBLIC -std=c++17 -Wall )

# Set the final name of the executable.
set_target_properties (neural_network PROPERTIES OUTPUT_NAME "neural_network_app")


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
