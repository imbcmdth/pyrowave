# WebGPU backend for PyroWave. Built from the top level with -DPYROWAVE_WEBGPU=ON, or
# standalone from this directory, which needs neither Granite nor Vulkan.
#
# Any implementation of the standard webgpu.h works. If the parent project already
# defines a `webgpu` target (Dawn, or a webgpu.h shim), that is used. Otherwise a
# prebuilt wgpu-native is expected in PYROWAVE_WEBGPU_NATIVE_DIR, which
# fetch_wgpu_native.cmake downloads.

cmake_minimum_required(VERSION 3.20)
project(pyrowave-webgpu LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYROWAVE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(PYROWAVE_WEBGPU_NATIVE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/wgpu-native
    CACHE PATH "Prebuilt wgpu-native (include/webgpu/webgpu.h and lib/).")
option(PYROWAVE_WEBGPU_TOOLS "Build the WebGPU encode and decode command line tools." ON)

if (NOT TARGET webgpu)
	find_path(PYROWAVE_WEBGPU_INCLUDE_DIR webgpu/webgpu.h
	          PATHS ${PYROWAVE_WEBGPU_NATIVE_DIR}/include NO_DEFAULT_PATH)
	find_library(PYROWAVE_WEBGPU_LIBRARY NAMES wgpu_native.dll.lib wgpu_native
	             PATHS ${PYROWAVE_WEBGPU_NATIVE_DIR}/lib NO_DEFAULT_PATH)
	if (NOT PYROWAVE_WEBGPU_INCLUDE_DIR OR NOT PYROWAVE_WEBGPU_LIBRARY)
		message(FATAL_ERROR "webgpu.h not found. Run `cmake -P webgpu/fetch_wgpu_native.cmake`, "
		                    "set PYROWAVE_WEBGPU_NATIVE_DIR, or define a `webgpu` target before this directory.")
	endif()

	add_library(webgpu INTERFACE)
	target_include_directories(webgpu INTERFACE ${PYROWAVE_WEBGPU_INCLUDE_DIR})
	target_link_libraries(webgpu INTERFACE ${PYROWAVE_WEBGPU_LIBRARY})
	# Enables the one wgpu-native specific fallback, see pyrowave_webgpu_common.cpp.
	target_compile_definitions(webgpu INTERFACE PYROWAVE_WEBGPU_WGPU_NATIVE=1)
	if (WIN32)
		set(PYROWAVE_WEBGPU_RUNTIME ${PYROWAVE_WEBGPU_NATIVE_DIR}/lib/wgpu_native.dll)
	endif()
endif()

set(WGSL_SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/subgroup.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/band_dispatch.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/dwt_common.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/dwt.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/idwt.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/wavelet_quant.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/wavelet_dequant.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/analyze_rate_control.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/analyze_rate_control_finalize.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/resolve_rate_control.wgsl
		${CMAKE_CURRENT_SOURCE_DIR}/shaders/block_packing.wgsl)

set(WGSL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/pyrowave_webgpu_shaders.hpp)
add_custom_command(OUTPUT ${WGSL_HEADER}
		COMMAND ${CMAKE_COMMAND} -DOUTPUT=${WGSL_HEADER} "-DINPUTS=${WGSL_SOURCES}"
		        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake
		DEPENDS ${WGSL_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake
		COMMENT "Embedding WGSL shaders")

# The Granite-free bitstream layer (block layout, packet parsing, packetization) is
# shared with the Metal port rather than duplicated.
add_library(pyrowave-webgpu STATIC
		pyrowave_webgpu.h
		pyrowave_webgpu_common.hpp pyrowave_webgpu_common.cpp
		pyrowave_webgpu_encoder.cpp
		pyrowave_webgpu_decoder.cpp
		${PYROWAVE_ROOT}/metal/pyrowave_bitstream.cpp ${PYROWAVE_ROOT}/metal/pyrowave_bitstream.hpp
		${WGSL_HEADER} ${WGSL_SOURCES})
target_include_directories(pyrowave-webgpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(pyrowave-webgpu PRIVATE ${PYROWAVE_ROOT}/metal ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(pyrowave-webgpu PUBLIC webgpu)
set_target_properties(pyrowave-webgpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (MSVC)
	target_compile_options(pyrowave-webgpu PRIVATE /W4 /wd4100 /wd4201 /wd4389 /D_CRT_SECURE_NO_WARNINGS)
else()
	target_compile_options(pyrowave-webgpu PRIVATE -Wall -Wextra -Wshadow -Wno-missing-field-initializers -fvisibility=hidden)
endif()

if (PYROWAVE_WEBGPU_TOOLS)
	add_library(pyrowave-webgpu-tool-utils STATIC ${PYROWAVE_ROOT}/yuv4mpeg.cpp ${PYROWAVE_ROOT}/yuv4mpeg.hpp)
	target_include_directories(pyrowave-webgpu-tool-utils PUBLIC ${PYROWAVE_ROOT})
	if (MSVC)
		target_compile_definitions(pyrowave-webgpu-tool-utils PRIVATE _CRT_SECURE_NO_WARNINGS)
	endif()

	foreach (tool encode decode)
		add_executable(pyrowave-webgpu-${tool} tools/pyrowave_webgpu_${tool}.cpp tools/pyrowave_file_format.hpp)
		target_link_libraries(pyrowave-webgpu-${tool} PRIVATE pyrowave-webgpu pyrowave-webgpu-tool-utils)
		if (MSVC)
			target_compile_definitions(pyrowave-webgpu-${tool} PRIVATE _CRT_SECURE_NO_WARNINGS)
		endif()
		if (PYROWAVE_WEBGPU_RUNTIME)
			add_custom_command(TARGET pyrowave-webgpu-${tool} POST_BUILD
					COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PYROWAVE_WEBGPU_RUNTIME} $<TARGET_FILE_DIR:pyrowave-webgpu-${tool}>)
		endif()
	endforeach()

	# Dumps the Vulkan encoder's wavelet pyramid for comparison with the WebGPU one.
	if (TARGET pyrowave AND TARGET granite-vulkan)
		add_executable(pyrowave-vulkan-dwt-dump tools/pyrowave_vulkan_dwt_dump.cpp)
		target_link_libraries(pyrowave-vulkan-dwt-dump PRIVATE pyrowave granite-vulkan pyrowave-webgpu-tool-utils)
		if (MSVC)
			target_compile_definitions(pyrowave-vulkan-dwt-dump PRIVATE _CRT_SECURE_NO_WARNINGS)
		endif()
	endif()

	# Encode and decode through the Vulkan C API (pyrowave.h) with the same file format,
	# for checking the two backends against each other. Only when built from the top
	# level, where the Vulkan library exists.
	if (TARGET pyrowave-shared)
		add_executable(pyrowave-vulkan-cli tools/pyrowave_vulkan_cli.cpp tools/pyrowave_file_format.hpp)
		target_link_libraries(pyrowave-vulkan-cli PRIVATE pyrowave-shared pyrowave-webgpu-tool-utils granite-volk-headers)
		target_include_directories(pyrowave-vulkan-cli PRIVATE ${PYROWAVE_ROOT})
		if (WIN32)
			add_custom_command(TARGET pyrowave-vulkan-cli POST_BUILD
					COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pyrowave-shared> $<TARGET_FILE_DIR:pyrowave-vulkan-cli>)
		endif()
		if (MSVC)
			target_compile_definitions(pyrowave-vulkan-cli PRIVATE _CRT_SECURE_NO_WARNINGS)
		endif()
	endif()
endif()
