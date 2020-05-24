cmake_minimum_required(VERSION 3.5)
project(RnNoise LANGUAGES C)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

# specify the cross compiler
#SET(CMAKE_C_COMPILER   /opt/gcc-armv7l-meego-linux-gnueabi-2016.01_linux/cross/bin/armv7l-meego-linux-gnueabi-gcc)
#SET(CMAKE_CXX_COMPILER /opt/gcc-armv7l-meego-linux-gnueabi-2016.01_linux/cross/bin/armv7l-meego-linux-gnueabi-g++)

# where is the target environment 
#SET(CMAKE_FIND_ROOT_PATH  /opt/gcc-armv7l-meego-linux-gnueabi-2016.01_linux/cross/armv7l-meego-linux-gnueabi/sys-root/)
#SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
#SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# specify the compiler flag
#SET(CMAKE_C_FLAGS  "-std=gnu99 -O3 -mfpu=neon")
#SET(CMAKE_CXX_FLAGS  "-std=c++11 -O3 -mfpu=neon")

add_subdirectory(src)
add_subdirectory(examples)

