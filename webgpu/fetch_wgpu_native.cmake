# Downloads a pinned prebuilt wgpu-native into webgpu/external/, which is ignored by git.
# wgpu-native is only used to run the WebGPU backend natively; the backend itself only
# includes the standard webgpu.h, so any implementation of that header will do.
#
#   cmake -P webgpu/fetch_wgpu_native.cmake
#
# Then configure with -DPYROWAVE_WEBGPU=ON. The CMakeLists.txt finds the result by
# default, or point PYROWAVE_WEBGPU_NATIVE_DIR somewhere else.

cmake_minimum_required(VERSION 3.20)

set(WGPU_NATIVE_VERSION v29.0.1.1)

if (CMAKE_HOST_WIN32)
	if ("$ENV{PROCESSOR_ARCHITECTURE}" STREQUAL "ARM64")
		set(WGPU_NATIVE_PLATFORM windows-aarch64-msvc)
		set(WGPU_NATIVE_SHA256 4a876421a8c1e5fe72f849b3722214280fe485cb1c56f77f8b0c82414be5b29f)
	else()
		set(WGPU_NATIVE_PLATFORM windows-x86_64-msvc)
		set(WGPU_NATIVE_SHA256 7e67d7445c42aeb85e30f88930fd8d7d83ee769e3390aeb1ada75ebf3cf78132)
	endif()
elseif (CMAKE_HOST_APPLE)
	set(WGPU_NATIVE_PLATFORM macos-aarch64)
	set(WGPU_NATIVE_SHA256 a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e)
else()
	set(WGPU_NATIVE_PLATFORM linux-x86_64)
	set(WGPU_NATIVE_SHA256 95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a)
endif()

set(WGPU_NATIVE_ARCHIVE wgpu-${WGPU_NATIVE_PLATFORM}-release.zip)
set(WGPU_NATIVE_URL https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_NATIVE_VERSION}/${WGPU_NATIVE_ARCHIVE})
set(WGPU_NATIVE_DEST ${CMAKE_CURRENT_LIST_DIR}/external/wgpu-native)

if (EXISTS ${WGPU_NATIVE_DEST}/wgpu-native-meta/wgpu-native-git-tag)
	file(READ ${WGPU_NATIVE_DEST}/wgpu-native-meta/wgpu-native-git-tag existing_tag)
	string(STRIP "${existing_tag}" existing_tag)
	if ("${existing_tag}" STREQUAL "${WGPU_NATIVE_VERSION}")
		message(STATUS "wgpu-native ${WGPU_NATIVE_VERSION} is already in ${WGPU_NATIVE_DEST}.")
		return()
	endif()
endif()

set(download ${CMAKE_CURRENT_LIST_DIR}/external/${WGPU_NATIVE_ARCHIVE})
message(STATUS "Downloading ${WGPU_NATIVE_URL}")
file(DOWNLOAD ${WGPU_NATIVE_URL} ${download}
     EXPECTED_HASH SHA256=${WGPU_NATIVE_SHA256}
     STATUS status)
list(GET status 0 status_code)
if (NOT status_code EQUAL 0)
	message(FATAL_ERROR "Download failed: ${status}")
endif()

file(REMOVE_RECURSE ${WGPU_NATIVE_DEST})
file(ARCHIVE_EXTRACT INPUT ${download} DESTINATION ${WGPU_NATIVE_DEST})
file(REMOVE ${download})
message(STATUS "Extracted wgpu-native ${WGPU_NATIVE_VERSION} (${WGPU_NATIVE_PLATFORM}) to ${WGPU_NATIVE_DEST}.")
