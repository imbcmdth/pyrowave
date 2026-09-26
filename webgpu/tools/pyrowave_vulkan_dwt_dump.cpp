// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT

// Validation aid for comparing the WebGPU backend with the Vulkan one stage by stage.
//
//   pyrowave-vulkan-dwt-dump dwt <input.y4m> <output.bin>
//       Runs the Vulkan encoder on the first frame and writes its wavelet pyramid as
//       float32, in the layout WaveletPyramid::dump() uses on the WebGPU side: level
//       by level, 12 layers of component * 4 + band, rows of the level's width.
//
//   pyrowave-vulkan-dwt-dump idwt <input.pyrowave> <output.bin> [frame]
//       Decodes one frame into R32F planes instead of R8, and writes the three planes
//       as float32, before any conversion to 8 bits.

#include "device.hpp"
#include "context.hpp"
#include "pyrowave_encoder.hpp"
#include "pyrowave_decoder.hpp"
#include "pyrowave_common.hpp"
#include "pyrowave_file_format.hpp"
#include "yuv4mpeg.hpp"
#include <math.h>
#include <string.h>
#include <vector>

using namespace Vulkan;

static float half_to_float(uint16_t h)
{
	int sign = (h >> 15) & 1;
	int exponent = (h >> 10) & 0x1f;
	int mantissa = h & 0x3ff;
	float v;
	if (exponent == 0)
		v = ldexpf(float(mantissa), -24);
	else if (exponent == 31)
		v = mantissa ? NAN : INFINITY;
	else
		v = ldexpf(float(mantissa + 1024), exponent - 25);
	return sign ? -v : v;
}

struct Readback
{
	BufferHandle buffer;
	unsigned width = 0, height = 0, layers = 0;
	bool fp16 = false;
};

static Readback copy_view(Device &device, CommandBuffer &cmd, const ImageView &view, unsigned layers)
{
	Readback r;
	r.width = view.get_view_width();
	r.height = view.get_view_height();
	r.layers = layers;
	r.fp16 = view.get_format() == VK_FORMAT_R16_SFLOAT;

	BufferCreateInfo info = {};
	info.domain = BufferDomain::CachedHost;
	info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	info.size = VkDeviceSize(r.width) * r.height * (r.fp16 ? 2 : 4) * layers;
	r.buffer = device.create_buffer(info);

	VkImageSubresourceLayers subresource = {
		VK_IMAGE_ASPECT_COLOR_BIT, view.get_create_info().base_level, view.get_create_info().base_layer, layers
	};
	cmd.copy_image_to_buffer(*r.buffer, view.get_image(), 0, {}, { r.width, r.height, 1 }, 0, 0, subresource);
	return r;
}

static void write_readback(Device &device, const Readback &r, FILE *out)
{
	auto *mapped = static_cast<const uint8_t *>(device.map_host_buffer(*r.buffer, MEMORY_ACCESS_READ_BIT));
	std::vector<float> values(size_t(r.width) * r.height * r.layers);
	for (size_t i = 0; i < values.size(); i++)
	{
		if (r.fp16)
		{
			uint16_t h;
			memcpy(&h, mapped + 2 * i, 2);
			values[i] = half_to_float(h);
		}
		else
			memcpy(&values[i], mapped + 4 * i, 4);
	}
	fwrite(values.data(), sizeof(float), values.size(), out);
}

static bool init_device(Context &ctx, Device &device)
{
	if (!Context::init_loader(nullptr))
		return false;
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0, CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT))
		return false;
	device.set_context(ctx);
	return true;
}

static int dump_dwt(const char *in_path, const char *out_path)
{
	YUV4MPEGFile input;
	if (!input.open_read(in_path) || !input.begin_frame())
		return EXIT_FAILURE;

	int width = input.get_width();
	int height = input.get_height();
	bool is_420 = YUV4MPEGFile::format_has_subsampling(input.get_format());
	auto chroma = is_420 ? PyroWave::ChromaSubsampling::Chroma420 : PyroWave::ChromaSubsampling::Chroma444;

	Context ctx;
	Device device;
	if (!init_device(ctx, device))
		return EXIT_FAILURE;

	ImageHandle planes[3];
	PyroWave::ViewBuffers views = {};
	for (int i = 0; i < 3; i++)
	{
		int w = i && is_420 ? width / 2 : width;
		int h = i && is_420 ? height / 2 : height;
		std::vector<uint8_t> data(size_t(w) * h);
		if (!input.read(data.data(), data.size()))
			return EXIT_FAILURE;
		ImageInitialData initial = { data.data(), uint32_t(w) };
		auto info = ImageCreateInfo::immutable_2d_image(w, h, VK_FORMAT_R8_UNORM);
		planes[i] = device.create_image(info, &initial);
		views.planes[i] = &planes[i]->get_view();
	}

	PyroWave::Encoder encoder;
	if (!encoder.init(&device, width, height, chroma))
		return EXIT_FAILURE;

	BufferCreateInfo buffer_info = {};
	buffer_info.domain = BufferDomain::Device;
	buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	buffer_info.size = encoder.get_meta_required_size();
	auto meta = device.create_buffer(buffer_info);
	buffer_info.size = 1024 * 1024 + encoder.get_meta_required_size();
	auto bitstream = device.create_buffer(buffer_info);

	PyroWave::Encoder::BitstreamBuffers buffers = {};
	buffers.meta.buffer = meta.get();
	buffers.meta.size = meta->get_create_info().size;
	buffers.bitstream.buffer = bitstream.get();
	buffers.bitstream.size = bitstream->get_create_info().size;
	buffers.target_size = 1024 * 1024;

	auto cmd = device.request_command_buffer();
	if (!encoder.encode(*cmd, views, buffers))
		return EXIT_FAILURE;

	cmd->barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

	Readback readbacks[PyroWave::DecompositionLevels][PyroWave::NumComponents];
	for (int level = 0; level < PyroWave::DecompositionLevels; level++)
		for (int c = 0; c < PyroWave::NumComponents; c++)
			readbacks[level][c] = copy_view(device, *cmd, encoder.get_wavelet_band(c, level), 4);

	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
	Fence fence;
	device.submit(cmd, &fence);
	fence->wait();

	FILE *out = fopen(out_path, "wb");
	if (!out)
		return EXIT_FAILURE;
	for (auto &level : readbacks)
		for (auto &r : level)
			write_readback(device, r, out);
	fclose(out);
	return EXIT_SUCCESS;
}

static int dump_idwt(const char *in_path, const char *out_path, int frame)
{
	FILE *in = fopen(in_path, "rb");
	PyroWaveFile::Header header;
	if (!in || !PyroWaveFile::read_header(in, header))
		return EXIT_FAILURE;

	bool is_420 = header.chroma == 0;
	Context ctx;
	Device device;
	if (!init_device(ctx, device))
		return EXIT_FAILURE;

	PyroWave::Decoder decoder;
	if (!decoder.init(&device, header.width, header.height,
	                  is_420 ? PyroWave::ChromaSubsampling::Chroma420 : PyroWave::ChromaSubsampling::Chroma444))
		return EXIT_FAILURE;

	std::vector<uint8_t> packet;
	int decoded = -1;
	while (decoded < frame)
	{
		bool ready = false;
		while (PyroWaveFile::read_packet(in, packet))
		{
			if (!decoder.push_packet(packet.data(), packet.size()))
				return EXIT_FAILURE;
			if (decoder.decode_is_ready(false))
			{
				ready = true;
				break;
			}
		}
		if (!ready)
			return EXIT_FAILURE;
		decoded++;
		if (decoded < frame)
			decoder.clear();
	}
	fclose(in);

	ImageHandle planes[3];
	PyroWave::ViewBuffers views = {};
	for (int i = 0; i < 3; i++)
	{
		int w = i && is_420 ? header.width / 2 : header.width;
		int h = i && is_420 ? header.height / 2 : header.height;
		auto info = ImageCreateInfo::immutable_2d_image(w, h, VK_FORMAT_R32_SFLOAT);
		info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		info.initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
		planes[i] = device.create_image(info);
		views.planes[i] = &planes[i]->get_view();
	}

	auto cmd = device.request_command_buffer();
	for (auto &plane : planes)
	{
		cmd->image_barrier(*plane, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		                   VK_PIPELINE_STAGE_2_COPY_BIT, 0,
		                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
	}
	if (!decoder.decode(*cmd, views))
		return EXIT_FAILURE;
	cmd->barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

	Readback readbacks[3];
	for (int i = 0; i < 3; i++)
		readbacks[i] = copy_view(device, *cmd, planes[i]->get_view(), 1);

	cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	             VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
	Fence fence;
	device.submit(cmd, &fence);
	fence->wait();

	FILE *out = fopen(out_path, "wb");
	if (!out)
		return EXIT_FAILURE;
	for (auto &r : readbacks)
		write_readback(device, r, out);
	fclose(out);
	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	if (argc >= 4 && strcmp(argv[1], "dwt") == 0)
		return dump_dwt(argv[2], argv[3]);
	if (argc >= 4 && strcmp(argv[1], "idwt") == 0)
		return dump_idwt(argv[2], argv[3], argc >= 5 ? atoi(argv[4]) : 0);

	fprintf(stderr, "Usage: pyrowave-vulkan-dwt-dump dwt <input.y4m> <output.bin>\n"
	                "       pyrowave-vulkan-dwt-dump idwt <input.pyrowave> <output.bin> [frame]\n");
	return EXIT_FAILURE;
}
