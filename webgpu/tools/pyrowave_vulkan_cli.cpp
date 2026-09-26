// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT

// Encode and decode through the Vulkan C API (pyrowave.h), with the same file format
// and timing output as the WebGPU tools, so the two backends can be checked against
// each other:
//
//   pyrowave-vulkan-cli encode <input.y4m> <output.pyrowave> <bytes_per_frame> [--frames N]
//   pyrowave-vulkan-cli decode <input.pyrowave> <output.y4m>
//
// encode.cpp and decode.cpp do the same through the C++ API, but they need the
// PYROWAVE_DEVEL build with its SDL and renderer dependencies.

#include "volk.h"
#include "pyrowave.h"
#include "pyrowave_file_format.hpp"
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace PyroWaveFile;

static void report_stats(pyrowave_device device)
{
	pyrowave_device_report_performance_stats(device, [](void *, const char *msg) {
		if (strstr(msg, "Memory Heap") == nullptr)
			printf("  GPU %s\n", msg);
	}, nullptr, true);
}

struct FileDeleter { void operator()(FILE *f) { if (f) fclose(f); } };

static int run_encode(int argc, char **argv)
{
	if (argc < 5)
		return EXIT_FAILURE;

	size_t bytes_per_frame = strtoul(argv[4], nullptr, 0);
	int max_frames = -1;
	for (int i = 5; i < argc; i++)
		if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
			max_frames = atoi(argv[++i]);

	YUV4MPEGFile input;
	if (!input.open_read(argv[2]) || YUV4MPEGFile::format_to_bytes_per_component(input.get_format()) != 1)
	{
		fprintf(stderr, "Failed to open %s as 8-bit y4m.\n", argv[2]);
		return EXIT_FAILURE;
	}

	int width = input.get_width();
	int height = input.get_height();
	bool is_420 = YUV4MPEGFile::format_has_subsampling(input.get_format());

	std::unique_ptr<FILE, FileDeleter> out(fopen(argv[3], "wb"));
	Header header;
	header.width = width;
	header.height = height;
	header.format = int32_t(input.get_format());
	header.chroma = is_420 ? 0 : 1;
	header.full_range = input.is_full_range();
	header.frame_rate_num = input.get_frame_rate_num();
	header.frame_rate_den = input.get_frame_rate_den();
	if (!out || !write_header(out.get(), header))
		return EXIT_FAILURE;

	pyrowave_device device;
	if (pyrowave_create_default_device(&device) != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "Failed to create Vulkan device.\n");
		return EXIT_FAILURE;
	}

	pyrowave_encoder_create_info info = {};
	info.device = device;
	info.width = width;
	info.height = height;
	info.chroma = is_420 ? PYROWAVE_CHROMA_SUBSAMPLING_420 : PYROWAVE_CHROMA_SUBSAMPLING_444;
	pyrowave_encoder encoder;
	if (pyrowave_encoder_create(&info, &encoder) != PYROWAVE_SUCCESS)
	{
		pyrowave_device_destroy(device);
		return EXIT_FAILURE;
	}

	int cw = is_420 ? width / 2 : width;
	int ch = is_420 ? height / 2 : height;
	std::vector<uint8_t> planes[3];
	planes[0].resize(size_t(width) * height);
	planes[1].resize(size_t(cw) * ch);
	planes[2].resize(size_t(cw) * ch);

	pyrowave_cpu_buffer cpu = {};
	cpu.width = width;
	cpu.height = height;
	cpu.format = is_420 ? PYROWAVE_CPU_BUFFER_FORMAT_YUV420P : PYROWAVE_CPU_BUFFER_FORMAT_YUV444P;
	for (int i = 0; i < 3; i++)
	{
		cpu.data[i] = planes[i].data();
		cpu.row_stride_in_bytes[i] = i == 0 ? size_t(width) : size_t(cw);
		cpu.plane_size_in_bytes[i] = planes[i].size();
	}

	pyrowave_rate_control rate = { bytes_per_frame };
	std::vector<uint8_t> packetized(bytes_per_frame + 64 * 1024);
	TimeStats submit_stats, wait_stats, total_stats;
	size_t total_bytes = 0;
	int frames = 0;

	while ((max_frames < 0 || frames < max_frames) && input.begin_frame())
	{
		bool ok = true;
		for (int i = 0; i < 3 && ok; i++)
			ok = input.read(planes[i].data(), planes[i].size());
		if (!ok)
			break;

		Stopwatch total;
		Stopwatch submit;
		if (pyrowave_encoder_encode_cpu_synchronous(encoder, &cpu, &rate) != PYROWAVE_SUCCESS)
			break;
		double submit_ms = submit.elapsed_ms();

		// The packet queries block on the GPU.
		Stopwatch wait;
		pyrowave_packet packet = {};
		size_t num_packets = 0;
		if (pyrowave_encoder_packetize(encoder, &packet, packetized.size(), &num_packets,
		                               packetized.data(), packetized.size()) != PYROWAVE_SUCCESS || num_packets != 1)
			break;
		double wait_ms = wait.elapsed_ms();
		double total_ms = total.elapsed_ms();

		if (!write_packet(out.get(), packetized.data() + packet.offset, uint32_t(packet.size)))
			break;

		if (frames > 0)
		{
			submit_stats.add(submit_ms);
			wait_stats.add(wait_ms);
			total_stats.add(total_ms);
		}
		else
		{
			// Drop the warmup frame from the GPU statistics too.
			pyrowave_device_report_performance_stats(device, [](void *, const char *) {}, nullptr, true);
		}

		total_bytes += packet.size;
		frames++;
	}

	printf("Encoded %d frames of %dx%d %s, %.1f bytes per frame on average (target %zu).\n",
	       frames, width, height, is_420 ? "4:2:0" : "4:4:4",
	       frames ? double(total_bytes) / frames : 0.0, bytes_per_frame);
	printf("Per frame, excluding the first:\n");
	submit_stats.print("upload + record + submit");
	wait_stats.print("wait + readback + packetize");
	total_stats.print("total");
	report_stats(device);

	pyrowave_encoder_destroy(encoder);
	pyrowave_device_destroy(device);
	return frames > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int run_decode(int argc, char **argv)
{
	if (argc < 4)
		return EXIT_FAILURE;

	std::unique_ptr<FILE, FileDeleter> in(fopen(argv[2], "rb"));
	Header header;
	if (!in || !read_header(in.get(), header))
		return EXIT_FAILURE;

	YUV4MPEGFile output;
	if (!output.open_write(argv[3], y4m_params(header)))
		return EXIT_FAILURE;

	pyrowave_device device;
	if (pyrowave_create_default_device(&device) != PYROWAVE_SUCCESS)
		return EXIT_FAILURE;

	bool is_420 = header.chroma == 0;
	pyrowave_decoder_create_info info = {};
	info.device = device;
	info.width = header.width;
	info.height = header.height;
	info.chroma = is_420 ? PYROWAVE_CHROMA_SUBSAMPLING_420 : PYROWAVE_CHROMA_SUBSAMPLING_444;
	pyrowave_decoder decoder;
	if (pyrowave_decoder_create(&info, &decoder) != PYROWAVE_SUCCESS)
	{
		pyrowave_device_destroy(device);
		return EXIT_FAILURE;
	}

	int width = header.width;
	int height = header.height;
	int cw = is_420 ? width / 2 : width;
	int ch = is_420 ? height / 2 : height;
	std::vector<uint8_t> planes[3];
	planes[0].resize(size_t(width) * height);
	planes[1].resize(size_t(cw) * ch);
	planes[2].resize(size_t(cw) * ch);

	pyrowave_cpu_buffer cpu = {};
	cpu.width = width;
	cpu.height = height;
	cpu.format = is_420 ? PYROWAVE_CPU_BUFFER_FORMAT_YUV420P : PYROWAVE_CPU_BUFFER_FORMAT_YUV444P;
	for (int i = 0; i < 3; i++)
	{
		cpu.data[i] = planes[i].data();
		cpu.row_stride_in_bytes[i] = i == 0 ? size_t(width) : size_t(cw);
		cpu.plane_size_in_bytes[i] = planes[i].size();
	}

	TimeStats decode_stats;
	std::vector<uint8_t> packet;
	int frames = 0;

	for (;;)
	{
		bool ready = false;
		while (read_packet(in.get(), packet))
		{
			if (pyrowave_decoder_push_packet(decoder, packet.data(), packet.size()) != PYROWAVE_SUCCESS)
				break;
			if (pyrowave_decoder_decode_is_ready(decoder, false))
			{
				ready = true;
				break;
			}
		}
		if (!ready)
			break;

		Stopwatch decode;
		if (pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &cpu) != PYROWAVE_SUCCESS)
			break;
		if (frames > 0)
			decode_stats.add(decode.elapsed_ms());
		else
			pyrowave_device_report_performance_stats(device, [](void *, const char *) {}, nullptr, true);

		if (!output.begin_frame())
			break;
		bool ok = true;
		for (int i = 0; i < 3 && ok; i++)
			ok = output.write(planes[i].data(), planes[i].size());
		if (!ok)
			break;
		frames++;
	}

	printf("Decoded %d frames of %dx%d %s.\n", frames, width, height, is_420 ? "4:2:0" : "4:4:4");
	printf("Per frame, excluding the first:\n");
	decode_stats.print("upload + decode + readback");
	report_stats(device);

	pyrowave_decoder_destroy(decoder);
	pyrowave_device_destroy(device);
	return frames > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv)
{
	if (argc >= 2 && strcmp(argv[1], "encode") == 0)
		return run_encode(argc, argv);
	if (argc >= 2 && strcmp(argv[1], "decode") == 0)
		return run_decode(argc, argv);

	fprintf(stderr, "Usage: pyrowave-vulkan-cli encode <input.y4m> <output.pyrowave> <bytes_per_frame> [--frames N]\n"
	                "       pyrowave-vulkan-cli decode <input.pyrowave> <output.y4m>\n");
	return EXIT_FAILURE;
}
