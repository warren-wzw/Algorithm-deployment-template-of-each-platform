# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wzw/00-Ascend/project/pico_det/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wzw/00-Ascend/project/pico_det/build/intermediates/soc

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/clipper.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/clipper.cpp.o: /home/wzw/00-Ascend/project/pico_det/src/clipper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wzw/00-Ascend/project/pico_det/build/intermediates/soc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/clipper.cpp.o"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/clipper.cpp.o -c /home/wzw/00-Ascend/project/pico_det/src/clipper.cpp

CMakeFiles/main.dir/clipper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/clipper.cpp.i"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wzw/00-Ascend/project/pico_det/src/clipper.cpp > CMakeFiles/main.dir/clipper.cpp.i

CMakeFiles/main.dir/clipper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/clipper.cpp.s"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wzw/00-Ascend/project/pico_det/src/clipper.cpp -o CMakeFiles/main.dir/clipper.cpp.s

CMakeFiles/main.dir/clipper.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/clipper.cpp.o.requires

CMakeFiles/main.dir/clipper.cpp.o.provides: CMakeFiles/main.dir/clipper.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/clipper.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/clipper.cpp.o.provides

CMakeFiles/main.dir/clipper.cpp.o.provides.build: CMakeFiles/main.dir/clipper.cpp.o


CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.o: /home/wzw/00-Ascend/project/pico_det/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wzw/00-Ascend/project/pico_det/build/intermediates/soc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/main.cpp.o"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.cpp.o -c /home/wzw/00-Ascend/project/pico_det/src/main.cpp

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wzw/00-Ascend/project/pico_det/src/main.cpp > CMakeFiles/main.dir/main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wzw/00-Ascend/project/pico_det/src/main.cpp -o CMakeFiles/main.dir/main.cpp.s

CMakeFiles/main.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/main.cpp.o.requires

CMakeFiles/main.dir/main.cpp.o.provides: CMakeFiles/main.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/main.cpp.o.provides

CMakeFiles/main.dir/main.cpp.o.provides.build: CMakeFiles/main.dir/main.cpp.o


CMakeFiles/main.dir/postprocess_op.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/postprocess_op.cpp.o: /home/wzw/00-Ascend/project/pico_det/src/postprocess_op.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wzw/00-Ascend/project/pico_det/build/intermediates/soc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/postprocess_op.cpp.o"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/postprocess_op.cpp.o -c /home/wzw/00-Ascend/project/pico_det/src/postprocess_op.cpp

CMakeFiles/main.dir/postprocess_op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/postprocess_op.cpp.i"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wzw/00-Ascend/project/pico_det/src/postprocess_op.cpp > CMakeFiles/main.dir/postprocess_op.cpp.i

CMakeFiles/main.dir/postprocess_op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/postprocess_op.cpp.s"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wzw/00-Ascend/project/pico_det/src/postprocess_op.cpp -o CMakeFiles/main.dir/postprocess_op.cpp.s

CMakeFiles/main.dir/postprocess_op.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/postprocess_op.cpp.o.requires

CMakeFiles/main.dir/postprocess_op.cpp.o.provides: CMakeFiles/main.dir/postprocess_op.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/postprocess_op.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/postprocess_op.cpp.o.provides

CMakeFiles/main.dir/postprocess_op.cpp.o.provides.build: CMakeFiles/main.dir/postprocess_op.cpp.o


CMakeFiles/main.dir/preprocess_op.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/preprocess_op.cpp.o: /home/wzw/00-Ascend/project/pico_det/src/preprocess_op.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wzw/00-Ascend/project/pico_det/build/intermediates/soc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/main.dir/preprocess_op.cpp.o"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/preprocess_op.cpp.o -c /home/wzw/00-Ascend/project/pico_det/src/preprocess_op.cpp

CMakeFiles/main.dir/preprocess_op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/preprocess_op.cpp.i"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wzw/00-Ascend/project/pico_det/src/preprocess_op.cpp > CMakeFiles/main.dir/preprocess_op.cpp.i

CMakeFiles/main.dir/preprocess_op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/preprocess_op.cpp.s"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wzw/00-Ascend/project/pico_det/src/preprocess_op.cpp -o CMakeFiles/main.dir/preprocess_op.cpp.s

CMakeFiles/main.dir/preprocess_op.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/preprocess_op.cpp.o.requires

CMakeFiles/main.dir/preprocess_op.cpp.o.provides: CMakeFiles/main.dir/preprocess_op.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/preprocess_op.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/preprocess_op.cpp.o.provides

CMakeFiles/main.dir/preprocess_op.cpp.o.provides.build: CMakeFiles/main.dir/preprocess_op.cpp.o


CMakeFiles/main.dir/utility.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/utility.cpp.o: /home/wzw/00-Ascend/project/pico_det/src/utility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wzw/00-Ascend/project/pico_det/build/intermediates/soc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/main.dir/utility.cpp.o"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/utility.cpp.o -c /home/wzw/00-Ascend/project/pico_det/src/utility.cpp

CMakeFiles/main.dir/utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/utility.cpp.i"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wzw/00-Ascend/project/pico_det/src/utility.cpp > CMakeFiles/main.dir/utility.cpp.i

CMakeFiles/main.dir/utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/utility.cpp.s"
	/opt/linux/x86-arm/aarch64-v01c01-linux-gnu-gcc/bin/aarch64-v01c01-linux-gnu-gcc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wzw/00-Ascend/project/pico_det/src/utility.cpp -o CMakeFiles/main.dir/utility.cpp.s

CMakeFiles/main.dir/utility.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/utility.cpp.o.requires

CMakeFiles/main.dir/utility.cpp.o.provides: CMakeFiles/main.dir/utility.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/utility.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/utility.cpp.o.provides

CMakeFiles/main.dir/utility.cpp.o.provides.build: CMakeFiles/main.dir/utility.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/clipper.cpp.o" \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/postprocess_op.cpp.o" \
"CMakeFiles/main.dir/preprocess_op.cpp.o" \
"CMakeFiles/main.dir/utility.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

/home/wzw/00-Ascend/project/pico_det/out/main: CMakeFiles/main.dir/clipper.cpp.o
/home/wzw/00-Ascend/project/pico_det/out/main: CMakeFiles/main.dir/main.cpp.o
/home/wzw/00-Ascend/project/pico_det/out/main: CMakeFiles/main.dir/postprocess_op.cpp.o
/home/wzw/00-Ascend/project/pico_det/out/main: CMakeFiles/main.dir/preprocess_op.cpp.o
/home/wzw/00-Ascend/project/pico_det/out/main: CMakeFiles/main.dir/utility.cpp.o
/home/wzw/00-Ascend/project/pico_det/out/main: CMakeFiles/main.dir/build.make
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_img_hash.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: /home/wzw/00-Ascend/project/pico_det/3rdparty/opencv/opencv-3.4.3/lib/libopencv_world.so
/home/wzw/00-Ascend/project/pico_det/out/main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wzw/00-Ascend/project/pico_det/build/intermediates/soc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable /home/wzw/00-Ascend/project/pico_det/out/main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: /home/wzw/00-Ascend/project/pico_det/out/main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/clipper.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/main.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/postprocess_op.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/preprocess_op.cpp.o.requires
CMakeFiles/main.dir/requires: CMakeFiles/main.dir/utility.cpp.o.requires

.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/wzw/00-Ascend/project/pico_det/build/intermediates/soc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wzw/00-Ascend/project/pico_det/src /home/wzw/00-Ascend/project/pico_det/src /home/wzw/00-Ascend/project/pico_det/build/intermediates/soc /home/wzw/00-Ascend/project/pico_det/build/intermediates/soc /home/wzw/00-Ascend/project/pico_det/build/intermediates/soc/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend
