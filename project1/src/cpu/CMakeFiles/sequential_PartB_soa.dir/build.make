# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /nfsmnt/121090869/CUHKSZ-CSC4005/project1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /nfsmnt/121090869/CUHKSZ-CSC4005/project1

# Include any dependencies generated for this target.
include src/cpu/CMakeFiles/sequential_PartB_soa.dir/depend.make

# Include the progress variables for this target.
include src/cpu/CMakeFiles/sequential_PartB_soa.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpu/CMakeFiles/sequential_PartB_soa.dir/flags.make

src/cpu/CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.o: src/cpu/CMakeFiles/sequential_PartB_soa.dir/flags.make
src/cpu/CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.o: src/cpu/sequential_PartB_soa.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/121090869/CUHKSZ-CSC4005/project1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/cpu/CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.o"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.o -c /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/sequential_PartB_soa.cpp

src/cpu/CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.i"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/sequential_PartB_soa.cpp > CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.i

src/cpu/CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.s"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/sequential_PartB_soa.cpp -o CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.s

src/cpu/CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.o: src/cpu/CMakeFiles/sequential_PartB_soa.dir/flags.make
src/cpu/CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.o: src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/121090869/CUHKSZ-CSC4005/project1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/cpu/CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.o"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.o -c /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/utils.cpp

src/cpu/CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.i"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/utils.cpp > CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.i

src/cpu/CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.s"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/utils.cpp -o CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.s

# Object files for target sequential_PartB_soa
sequential_PartB_soa_OBJECTS = \
"CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.o" \
"CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.o"

# External object files for target sequential_PartB_soa
sequential_PartB_soa_EXTERNAL_OBJECTS =

src/cpu/sequential_PartB_soa: src/cpu/CMakeFiles/sequential_PartB_soa.dir/sequential_PartB_soa.cpp.o
src/cpu/sequential_PartB_soa: src/cpu/CMakeFiles/sequential_PartB_soa.dir/__/utils.cpp.o
src/cpu/sequential_PartB_soa: src/cpu/CMakeFiles/sequential_PartB_soa.dir/build.make
src/cpu/sequential_PartB_soa: src/cpu/CMakeFiles/sequential_PartB_soa.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/121090869/CUHKSZ-CSC4005/project1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable sequential_PartB_soa"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sequential_PartB_soa.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cpu/CMakeFiles/sequential_PartB_soa.dir/build: src/cpu/sequential_PartB_soa

.PHONY : src/cpu/CMakeFiles/sequential_PartB_soa.dir/build

src/cpu/CMakeFiles/sequential_PartB_soa.dir/clean:
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && $(CMAKE_COMMAND) -P CMakeFiles/sequential_PartB_soa.dir/cmake_clean.cmake
.PHONY : src/cpu/CMakeFiles/sequential_PartB_soa.dir/clean

src/cpu/CMakeFiles/sequential_PartB_soa.dir/depend:
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfsmnt/121090869/CUHKSZ-CSC4005/project1 /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu /nfsmnt/121090869/CUHKSZ-CSC4005/project1 /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/CMakeFiles/sequential_PartB_soa.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpu/CMakeFiles/sequential_PartB_soa.dir/depend
