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
include src/cpu/CMakeFiles/mpi_PartB.dir/depend.make

# Include the progress variables for this target.
include src/cpu/CMakeFiles/mpi_PartB.dir/progress.make

# Include the compile flags for this target's objects.
include src/cpu/CMakeFiles/mpi_PartB.dir/flags.make

src/cpu/CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.o: src/cpu/CMakeFiles/mpi_PartB.dir/flags.make
src/cpu/CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.o: src/cpu/mpi_PartB.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/121090869/CUHKSZ-CSC4005/project1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/cpu/CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.o"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.o -c /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/mpi_PartB.cpp

src/cpu/CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.i"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/mpi_PartB.cpp > CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.i

src/cpu/CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.s"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/mpi_PartB.cpp -o CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.s

src/cpu/CMakeFiles/mpi_PartB.dir/__/utils.cpp.o: src/cpu/CMakeFiles/mpi_PartB.dir/flags.make
src/cpu/CMakeFiles/mpi_PartB.dir/__/utils.cpp.o: src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/nfsmnt/121090869/CUHKSZ-CSC4005/project1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/cpu/CMakeFiles/mpi_PartB.dir/__/utils.cpp.o"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mpi_PartB.dir/__/utils.cpp.o -c /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/utils.cpp

src/cpu/CMakeFiles/mpi_PartB.dir/__/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mpi_PartB.dir/__/utils.cpp.i"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/utils.cpp > CMakeFiles/mpi_PartB.dir/__/utils.cpp.i

src/cpu/CMakeFiles/mpi_PartB.dir/__/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mpi_PartB.dir/__/utils.cpp.s"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && /opt/rh/devtoolset-7/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/utils.cpp -o CMakeFiles/mpi_PartB.dir/__/utils.cpp.s

# Object files for target mpi_PartB
mpi_PartB_OBJECTS = \
"CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.o" \
"CMakeFiles/mpi_PartB.dir/__/utils.cpp.o"

# External object files for target mpi_PartB
mpi_PartB_EXTERNAL_OBJECTS =

src/cpu/mpi_PartB: src/cpu/CMakeFiles/mpi_PartB.dir/mpi_PartB.cpp.o
src/cpu/mpi_PartB: src/cpu/CMakeFiles/mpi_PartB.dir/__/utils.cpp.o
src/cpu/mpi_PartB: src/cpu/CMakeFiles/mpi_PartB.dir/build.make
src/cpu/mpi_PartB: /usr/lib64/mpich-3.2/lib/libmpicxx.so
src/cpu/mpi_PartB: /usr/lib64/mpich-3.2/lib/libmpi.so
src/cpu/mpi_PartB: src/cpu/CMakeFiles/mpi_PartB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/nfsmnt/121090869/CUHKSZ-CSC4005/project1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable mpi_PartB"
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_PartB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cpu/CMakeFiles/mpi_PartB.dir/build: src/cpu/mpi_PartB

.PHONY : src/cpu/CMakeFiles/mpi_PartB.dir/build

src/cpu/CMakeFiles/mpi_PartB.dir/clean:
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu && $(CMAKE_COMMAND) -P CMakeFiles/mpi_PartB.dir/cmake_clean.cmake
.PHONY : src/cpu/CMakeFiles/mpi_PartB.dir/clean

src/cpu/CMakeFiles/mpi_PartB.dir/depend:
	cd /nfsmnt/121090869/CUHKSZ-CSC4005/project1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /nfsmnt/121090869/CUHKSZ-CSC4005/project1 /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu /nfsmnt/121090869/CUHKSZ-CSC4005/project1 /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu /nfsmnt/121090869/CUHKSZ-CSC4005/project1/src/cpu/CMakeFiles/mpi_PartB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cpu/CMakeFiles/mpi_PartB.dir/depend

