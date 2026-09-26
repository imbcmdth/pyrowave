// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT

// Decodes a file written by encode.cpp, pyrowave-webgpu-encode or pyrowave-vulkan-cli
// with the WebGPU backend, into y4m.
//
//   pyrowave-webgpu-decode <input.pyrowave> <output.y4m> [--timestamps]

#include "pyrowave_webgpu.h"
#include "pyrowave_file_format.hpp"
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace PyroWaveFile;

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "Usage: pyrowave-webgpu-decode <input.pyrowave> <output.y4m> [--timestamps]\n");
		return EXIT_FAILURE;
	}

	bool timestamps = false;
	for (int i = 3; i < argc; i++)
	{
		if (strcmp(argv[i], "--timestamps") == 0)
			timestamps = true;
		else
		{
			fprintf(stderr, "Unknown argument %s\n", argv[i]);
			return EXIT_FAILURE;
		}
	}

	struct FileDeleter { void operator()(FILE *f) { if (f) fclose(f); } };
	std::unique_ptr<FILE, FileDeleter> in(fopen(argv[1], "rb"));
	Header header;
	if (!in || !read_header(in.get(), header))
	{
		fprintf(stderr, "Failed to read %s.\n", argv[1]);
		return EXIT_FAILURE;
	}

	auto format = YUV4MPEGFile::Format(header.format);
	if (YUV4MPEGFile::format_to_bytes_per_component(format) != 1)
	{
		fprintf(stderr, "Only 8-bit output is supported.\n");
		return EXIT_FAILURE;
	}

	YUV4MPEGFile output;
	if (!output.open_write(argv[2], y4m_params(header)))
	{
		fprintf(stderr, "Failed to open %s.\n", argv[2]);
		return EXIT_FAILURE;
	}

	pyrowave_webgpu_device_create_info device_info = {};
	device_info.enable_timestamps = timestamps;
	pyrowave_webgpu_device device;
	if (pyrowave_webgpu_device_create(&device_info, &device) != PYROWAVE_WEBGPU_SUCCESS)
	{
		fprintf(stderr, "Failed to create WebGPU device.\n");
		return EXIT_FAILURE;
	}

	bool is_420 = header.chroma == 0;
	pyrowave_webgpu_decoder_create_info decoder_info = {};
	decoder_info.device = device;
	decoder_info.width = header.width;
	decoder_info.height = header.height;
	decoder_info.chroma = is_420 ? PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_420 : PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_444;
	pyrowave_webgpu_decoder decoder;
	auto result = pyrowave_webgpu_decoder_create(&decoder_info, &decoder);
	if (result != PYROWAVE_WEBGPU_SUCCESS)
	{
		fprintf(stderr, "Failed to create decoder: %s\n", pyrowave_webgpu_result_to_string(result));
		pyrowave_webgpu_device_destroy(device);
		return EXIT_FAILURE;
	}

	int width = header.width;
	int height = header.height;
	int chroma_width = is_420 ? width / 2 : width;
	int chroma_height = is_420 ? height / 2 : height;
	std::vector<uint8_t> planes[3];
	planes[0].resize(size_t(width) * height);
	planes[1].resize(size_t(chroma_width) * chroma_height);
	planes[2].resize(size_t(chroma_width) * chroma_height);

	pyrowave_webgpu_cpu_buffer cpu = {};
	cpu.width = width;
	cpu.height = height;
	cpu.format = is_420 ? PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV420P : PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV444P;
	for (int i = 0; i < 3; i++)
	{
		cpu.data[i] = planes[i].data();
		cpu.row_stride_in_bytes[i] = i == 0 ? size_t(width) : size_t(chroma_width);
		cpu.plane_size_in_bytes[i] = planes[i].size();
	}

	TimeStats submit_stats, wait_stats, read_stats, total_stats, dequant_stats, idwt_stats, gpu_total_stats;
	std::vector<uint8_t> packet;
	int frames = 0;

	for (;;)
	{
		bool ready = false;
		while (read_packet(in.get(), packet))
		{
			if (pyrowave_webgpu_decoder_push_packet(decoder, packet.data(), packet.size()) != PYROWAVE_WEBGPU_SUCCESS)
			{
				fprintf(stderr, "Corrupt packet.\n");
				break;
			}
			if (pyrowave_webgpu_decoder_decode_is_ready(decoder, false))
			{
				ready = true;
				break;
			}
		}

		if (!ready)
			break;

		Stopwatch total;
		Stopwatch submit;
		result = pyrowave_webgpu_decoder_decode_submit(decoder);
		double submit_ms = submit.elapsed_ms();
		Stopwatch wait;
		if (result == PYROWAVE_WEBGPU_SUCCESS)
			result = pyrowave_webgpu_decoder_decode_poll(decoder, true);
		double wait_ms = wait.elapsed_ms();
		Stopwatch read;
		if (result == PYROWAVE_WEBGPU_SUCCESS)
			result = pyrowave_webgpu_decoder_decode_read(decoder, &cpu);
		double read_ms = read.elapsed_ms();
		double total_ms = total.elapsed_ms();

		if (result != PYROWAVE_WEBGPU_SUCCESS)
		{
			fprintf(stderr, "Decode failed: %s\n", pyrowave_webgpu_result_to_string(result));
			break;
		}

		if (frames > 0)
		{
			submit_stats.add(submit_ms);
			wait_stats.add(wait_ms);
			read_stats.add(read_ms);
			total_stats.add(total_ms);

			pyrowave_webgpu_timings timings;
			if (timestamps && pyrowave_webgpu_decoder_get_timings(decoder, &timings) == PYROWAVE_WEBGPU_SUCCESS &&
			    timings.total_ns)
			{
				dequant_stats.add(double(timings.stage_ns[0]) * 1e-6);
				idwt_stats.add(double(timings.stage_ns[1]) * 1e-6);
				gpu_total_stats.add(double(timings.total_ns) * 1e-6);
			}
		}

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
	submit_stats.print("upload + record + submit");
	wait_stats.print("wait for GPU + readback");
	read_stats.print("copy out of mapped buffer");
	total_stats.print("total");
	dequant_stats.print("GPU dequantize");
	idwt_stats.print("GPU iDWT");
	gpu_total_stats.print("GPU total (first to last pass)");

	pyrowave_webgpu_decoder_destroy(decoder);
	pyrowave_webgpu_device_destroy(device);
	return frames > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
