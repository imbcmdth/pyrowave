// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT

// Encodes a y4m file with the WebGPU backend into the same container as encode.cpp.
//
//   pyrowave-webgpu-encode <input.y4m> <output.pyrowave> <bytes_per_frame>
//       [--frames N] [--timestamps] [--transfer-bench]
//
// Prints per frame timings at the end. --timestamps adds GPU time per stage.
// --transfer-bench additionally measures a frame upload and a bitstream sized
// readback on their own.

#include "pyrowave_webgpu.h"
#include "pyrowave_file_format.hpp"
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace PyroWaveFile;

static void measure_transfers(pyrowave_webgpu_device device, int width, int height, bool is_420,
                              size_t readback_size, int iterations);

int main(int argc, char **argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "Usage: pyrowave-webgpu-encode <input.y4m> <output.pyrowave> <bytes_per_frame> "
		                "[--frames N] [--timestamps] [--transfer-bench]\n");
		return EXIT_FAILURE;
	}

	const char *in_path = argv[1];
	const char *out_path = argv[2];
	size_t bytes_per_frame = strtoul(argv[3], nullptr, 0);
	int max_frames = -1;
	bool timestamps = false;
	bool transfer_bench = false;

	for (int i = 4; i < argc; i++)
	{
		if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
			max_frames = atoi(argv[++i]);
		else if (strcmp(argv[i], "--timestamps") == 0)
			timestamps = true;
		else if (strcmp(argv[i], "--transfer-bench") == 0)
			transfer_bench = true;
		else
		{
			fprintf(stderr, "Unknown argument %s\n", argv[i]);
			return EXIT_FAILURE;
		}
	}

	YUV4MPEGFile input;
	if (!input.open_read(in_path))
	{
		fprintf(stderr, "Failed to open %s.\n", in_path);
		return EXIT_FAILURE;
	}

	if (YUV4MPEGFile::format_to_bytes_per_component(input.get_format()) != 1)
	{
		fprintf(stderr, "Only 8-bit input is supported.\n");
		return EXIT_FAILURE;
	}

	int width = input.get_width();
	int height = input.get_height();
	bool is_420 = YUV4MPEGFile::format_has_subsampling(input.get_format());

	struct FileDeleter { void operator()(FILE *f) { if (f) fclose(f); } };
	std::unique_ptr<FILE, FileDeleter> out(fopen(out_path, "wb"));
	if (!out)
	{
		fprintf(stderr, "Failed to open %s.\n", out_path);
		return EXIT_FAILURE;
	}

	Header header;
	header.width = width;
	header.height = height;
	header.format = int32_t(input.get_format());
	header.chroma = is_420 ? 0 : 1;
	header.full_range = input.is_full_range();
	header.frame_rate_num = input.get_frame_rate_num();
	header.frame_rate_den = input.get_frame_rate_den();
	if (!write_header(out.get(), header))
		return EXIT_FAILURE;

	pyrowave_webgpu_device_create_info device_info = {};
	device_info.enable_timestamps = timestamps;
	pyrowave_webgpu_device device;
	if (pyrowave_webgpu_device_create(&device_info, &device) != PYROWAVE_WEBGPU_SUCCESS)
	{
		fprintf(stderr, "Failed to create WebGPU device.\n");
		return EXIT_FAILURE;
	}

	pyrowave_webgpu_encoder_create_info encoder_info = {};
	encoder_info.device = device;
	encoder_info.width = width;
	encoder_info.height = height;
	encoder_info.chroma = is_420 ? PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_420 : PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_444;
	pyrowave_webgpu_encoder encoder;
	auto result = pyrowave_webgpu_encoder_create(&encoder_info, &encoder);
	if (result != PYROWAVE_WEBGPU_SUCCESS)
	{
		fprintf(stderr, "Failed to create encoder: %s\n", pyrowave_webgpu_result_to_string(result));
		pyrowave_webgpu_device_destroy(device);
		return EXIT_FAILURE;
	}

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

	pyrowave_webgpu_rate_control rate = { bytes_per_frame };
	std::vector<uint8_t> packetized(bytes_per_frame + 64 * 1024);

	TimeStats submit_stats, wait_stats, packetize_stats, total_stats;
	TimeStats stage_stats[5], gpu_total_stats;
	static const char *const stage_names[5] = { "GPU DWT", "GPU quantize", "GPU analyze", "GPU resolve", "GPU packing" };
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
		result = pyrowave_webgpu_encoder_encode_cpu(encoder, &cpu, &rate);
		if (result != PYROWAVE_WEBGPU_SUCCESS)
		{
			fprintf(stderr, "Encode failed: %s\n", pyrowave_webgpu_result_to_string(result));
			break;
		}
		double submit_ms = submit.elapsed_ms();

		Stopwatch wait;
		if (pyrowave_webgpu_encoder_poll(encoder, true) != PYROWAVE_WEBGPU_SUCCESS)
		{
			fprintf(stderr, "Readback failed.\n");
			break;
		}
		double wait_ms = wait.elapsed_ms();

		Stopwatch pack;
		pyrowave_webgpu_packet packet = {};
		size_t num_packets = 0;
		pyrowave_webgpu_encoder_packetize(encoder, &packet, packetized.size(), &num_packets,
		                                  packetized.data(), packetized.size());
		double pack_ms = pack.elapsed_ms();
		double total_ms = total.elapsed_ms();

		if (num_packets != 1)
		{
			fprintf(stderr, "Expected a single packet, got %zu.\n", num_packets);
			break;
		}

		if (!write_packet(out.get(), packetized.data() + packet.offset, uint32_t(packet.size)))
			break;

		// The first frame includes pipeline warmup, skip it in the statistics.
		if (frames > 0)
		{
			submit_stats.add(submit_ms);
			wait_stats.add(wait_ms);
			packetize_stats.add(pack_ms);
			total_stats.add(total_ms);

			pyrowave_webgpu_timings timings;
			if (timestamps && pyrowave_webgpu_encoder_get_timings(encoder, &timings) == PYROWAVE_WEBGPU_SUCCESS &&
			    timings.total_ns)
			{
				for (int i = 0; i < timings.num_stages && i < 5; i++)
					stage_stats[i].add(double(timings.stage_ns[i]) * 1e-6);
				gpu_total_stats.add(double(timings.total_ns) * 1e-6);
			}
		}

		total_bytes += packet.size;
		frames++;
	}

	printf("Encoded %d frames of %dx%d %s, %.1f bytes per frame on average (target %zu).\n",
	       frames, width, height, is_420 ? "4:2:0" : "4:4:4",
	       frames ? double(total_bytes) / frames : 0.0, bytes_per_frame);
	printf("Per frame, excluding the first:\n");
	submit_stats.print("upload + record + submit");
	wait_stats.print("wait for GPU + readback");
	packetize_stats.print("packetize");
	total_stats.print("total");
	for (int i = 0; i < 5; i++)
		stage_stats[i].print(stage_names[i]);
	gpu_total_stats.print("GPU total (first to last pass)");

	if (transfer_bench)
		measure_transfers(device, width, height, is_420, bytes_per_frame, 100);

	pyrowave_webgpu_encoder_destroy(encoder);
	pyrowave_webgpu_device_destroy(device);
	return frames > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

namespace
{
struct Flag
{
	bool done = false;
};

void on_work_done(WGPUQueueWorkDoneStatus, WGPUStringView, void *userdata1, void *)
{
	static_cast<Flag *>(userdata1)->done = true;
}

void on_mapped(WGPUMapAsyncStatus, WGPUStringView, void *userdata1, void *)
{
	static_cast<Flag *>(userdata1)->done = true;
}

void spin(WGPUInstance instance, const Flag &flag)
{
	while (!flag.done)
		wgpuInstanceProcessEvents(instance);
}
}

// Measures what moving a frame through CPU memory costs on its own: uploading one
// input frame with wgpuQueueWriteTexture, and reading back a buffer the size of the
// encoder's result with a copy and wgpuBufferMapAsync. Wall clock, round trip.
static void measure_transfers(pyrowave_webgpu_device device, int width, int height, bool is_420,
                              size_t readback_size, int iterations)
{
	WGPUInstance instance;
	WGPUDevice dev;
	pyrowave_webgpu_device_get_handles(device, &instance, nullptr, &dev);
	WGPUQueue queue = wgpuDeviceGetQueue(dev);

	int cw = is_420 ? width / 2 : width;
	int ch = is_420 ? height / 2 : height;
	int dims[3][2] = { { width, height }, { cw, ch }, { cw, ch } };
	WGPUTexture textures[3];
	std::vector<uint8_t> data(size_t(width) * height, 0x80);

	for (int i = 0; i < 3; i++)
	{
		WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
		desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
		desc.size = { uint32_t(dims[i][0]), uint32_t(dims[i][1]), 1 };
		desc.format = WGPUTextureFormat_R8Unorm;
		textures[i] = wgpuDeviceCreateTexture(dev, &desc);
	}

	WGPUBufferDescriptor buf_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
	buf_desc.size = (readback_size + 3) & ~size_t(3);
	buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
	WGPUBuffer src = wgpuDeviceCreateBuffer(dev, &buf_desc);
	buf_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
	WGPUBuffer dst = wgpuDeviceCreateBuffer(dev, &buf_desc);

	TimeStats upload_stats, readback_stats;
	for (int iter = 0; iter < iterations; iter++)
	{
		Stopwatch upload;
		for (int i = 0; i < 3; i++)
		{
			WGPUTexelCopyTextureInfo tex = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
			tex.texture = textures[i];
			WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
			layout.bytesPerRow = uint32_t(dims[i][0]);
			layout.rowsPerImage = uint32_t(dims[i][1]);
			WGPUExtent3D extent = { uint32_t(dims[i][0]), uint32_t(dims[i][1]), 1 };
			wgpuQueueWriteTexture(queue, &tex, data.data(), size_t(dims[i][0]) * dims[i][1], &layout, &extent);
		}
		Flag uploaded;
		WGPUQueueWorkDoneCallbackInfo done_info = {};
		done_info.mode = WGPUCallbackMode_AllowProcessEvents;
		done_info.callback = on_work_done;
		done_info.userdata1 = &uploaded;
		wgpuQueueOnSubmittedWorkDone(queue, done_info);
		spin(instance, uploaded);
		upload_stats.add(upload.elapsed_ms());

		Stopwatch readback;
		WGPUCommandEncoder cmd = wgpuDeviceCreateCommandEncoder(dev, nullptr);
		wgpuCommandEncoderCopyBufferToBuffer(cmd, src, 0, dst, 0, buf_desc.size);
		WGPUCommandBuffer cb = wgpuCommandEncoderFinish(cmd, nullptr);
		wgpuQueueSubmit(queue, 1, &cb);
		wgpuCommandBufferRelease(cb);
		wgpuCommandEncoderRelease(cmd);
		Flag mapped;
		WGPUBufferMapCallbackInfo map_info = {};
		map_info.mode = WGPUCallbackMode_AllowProcessEvents;
		map_info.callback = on_mapped;
		map_info.userdata1 = &mapped;
		wgpuBufferMapAsync(dst, WGPUMapMode_Read, 0, size_t(buf_desc.size), map_info);
		spin(instance, mapped);
		volatile uint8_t sink = static_cast<const uint8_t *>(wgpuBufferGetConstMappedRange(dst, 0, size_t(buf_desc.size)))[0];
		(void)sink;
		wgpuBufferUnmap(dst);
		readback_stats.add(readback.elapsed_ms());
	}

	printf("Transfers on their own (%d iterations, wall clock round trip):\n", iterations);
	upload_stats.print("upload one input frame");
	char label[64];
	snprintf(label, sizeof(label), "read back %zu bytes", size_t(buf_desc.size));
	readback_stats.print(label);

	for (auto *t : textures)
		wgpuTextureRelease(t);
	wgpuBufferRelease(src);
	wgpuBufferRelease(dst);
	wgpuQueueRelease(queue);
}
