// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT

// WebGPU backend for the PyroWave decoder. Mirrors the compute path of
// pyrowave_decoder.cpp: dequant -> iDWT. Packet parsing is the shared, API free
// BitstreamParser.
//
// The decoded planes land in one storage buffer of packed 8-bit samples, with rows
// padded to the aligned size, and are read back from there.

#include "pyrowave_webgpu_common.hpp"
#include <algorithm>
#include <new>

using namespace PyroWave;
using namespace PyroWave::WebGPU;

namespace
{
struct DequantRegisters
{
	int32_t resolution[2];
	int32_t output_layer;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	uint32_t store_fp16;
	uint32_t padding[2];
};
static_assert(sizeof(DequantRegisters) == 32, "Must match Registers in the WGSL.");

struct IDWTRegisters
{
	int32_t resolution[2];
	float inv_resolution[2];
	uint32_t store_fp16;
	uint32_t output_offset;
	uint32_t output_stride;
	uint32_t output_rows;
	uint32_t padding[4];
};
static_assert(sizeof(IDWTRegisters) == 48, "Must match Registers in the WGSL.");

enum Stage
{
	STAGE_DEQUANT,
	STAGE_IDWT,
	STAGE_COUNT
};

}

struct pyrowave_webgpu_decoder_opaque
{
	pyrowave_webgpu_device device = nullptr;
	BlockLayout layout;
	BitstreamParser parser;
	WaveletPyramid pyramid;

	WGPUBuffer dequant_offset_buffer = nullptr;
	WGPUBuffer payload_buffer = nullptr;
	uint64_t payload_buffer_size = 0;

	// Decoded planes, 8-bit samples packed four to a word.
	WGPUBuffer output_buffer = nullptr;
	uint64_t output_size = 0;
	uint32_t plane_offset[3] = {};
	uint32_t plane_stride[3] = {};

	WGPUBuffer uniform_buffer = nullptr;
	std::vector<uint8_t> uniform_data;

	std::vector<Dispatch> stages[STAGE_COUNT];
	// One dequant dispatch per level, covering its components and bands.
	BandTable dequant_bands[DecompositionLevels];

	StageTimer timer;
	Readback readback;
	uint64_t readback_timestamp_offset = 0;
	bool has_frame = false;

	bool init(pyrowave_webgpu_device device, int width, int height, ChromaSubsampling chroma);
	void release();
	bool plan_idwt(uint32_t &slot);
	bool plan_dequant();
	bool rebuild_dequant_groups();

	int plane_width(int plane) const;
	int plane_height(int plane) const;
	int aligned_plane_width(int plane) const;
	int aligned_plane_height(int plane) const;
};

int pyrowave_webgpu_decoder_opaque::plane_width(int plane) const
{
	return plane != 0 && layout.chroma == ChromaSubsampling::Chroma420 ? layout.width / 2 : layout.width;
}

int pyrowave_webgpu_decoder_opaque::plane_height(int plane) const
{
	return plane != 0 && layout.chroma == ChromaSubsampling::Chroma420 ? layout.height / 2 : layout.height;
}

int pyrowave_webgpu_decoder_opaque::aligned_plane_width(int plane) const
{
	return plane != 0 && layout.chroma == ChromaSubsampling::Chroma420 ? layout.aligned_width / 2 : layout.aligned_width;
}

int pyrowave_webgpu_decoder_opaque::aligned_plane_height(int plane) const
{
	return plane != 0 && layout.chroma == ChromaSubsampling::Chroma420 ? layout.aligned_height / 2 : layout.aligned_height;
}

bool pyrowave_webgpu_decoder_opaque::plan_dequant()
{
	// Mirrors Decoder::Impl::dequant(), with every band of a level in one dispatch,
	// see shaders/band_dispatch.wgsl. Levels stay separate because a storage texture
	// binding covers one mip level. The bind groups reference the payload buffer,
	// which grows with the frames, so they are created by rebuild_dequant_groups().
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				DequantRegisters regs = {};
				regs.resolution[0] = layout.level_width(level);
				regs.resolution[1] = layout.level_height(level);
				regs.output_layer = NumFrequencyBandsPerLevel * component + band;
				regs.block_offset_32x32 = layout.block_meta[component][level][band].block_offset_32x32;
				regs.block_stride_32x32 = layout.block_meta[component][level][band].block_stride_32x32;
				regs.store_fp16 = device->precision == 1 && level < WaveletFP16Levels;
				dequant_bands[level].add(regs, uint32_t(regs.resolution[0] + 31) / 32,
				                         uint32_t(regs.resolution[1] + 31) / 32);
			}
		}

		if (!dequant_bands[level].create(device, "dequant-bands"))
			return false;
		stages[STAGE_DEQUANT].push_back(dequant_bands[level].dispatch(device->dequant.pipeline, nullptr));
	}

	return true;
}

bool pyrowave_webgpu_decoder_opaque::rebuild_dequant_groups()
{
	auto &dispatches = stages[STAGE_DEQUANT];
	for (int level = 0; level < DecompositionLevels; level++)
	{
		auto &d = dispatches[level];
		if (d.bind_group)
			wgpuBindGroupRelease(d.bind_group);

		const BindingResource resources[] = {
			dequant_bands[level].bind_registers(),
			bind_view(1, pyramid.level_views[level]),
			bind_buffer(2, dequant_offset_buffer),
			bind_buffer(3, payload_buffer),
			dequant_bands[level].bind_ranges(),
		};
		d.bind_group = create_bind_group(device, device->dequant, resources, 5);
		if (!d.bind_group)
			return false;
	}
	return true;
}

bool pyrowave_webgpu_decoder_opaque::plan_idwt(uint32_t &slot)
{
	// Mirrors Decoder::Impl::idwt().
	auto &dispatches = stages[STAGE_IDWT];
	const bool is_420 = layout.chroma == ChromaSubsampling::Chroma420;

	for (int input_level = DecompositionLevels - 1; input_level >= 0; input_level--)
	{
		IDWTRegisters regs = {};
		// Transposed.
		regs.resolution[0] = layout.level_height(input_level);
		regs.resolution[1] = layout.level_width(input_level);
		regs.inv_resolution[0] = 1.0f / float(regs.resolution[0]);
		regs.inv_resolution[1] = 1.0f / float(regs.resolution[1]);

		for (int c = 0; c < NumComponents; c++)
		{
			bool final_output = input_level == 0 || (is_420 && c != 0 && input_level == 1);
			if (input_level == 0 && is_420 && c != 0)
				continue;

			IDWTRegisters component_regs = regs;
			Dispatch d = {};
			d.x = uint32_t(regs.resolution[0] + 15) / 16;
			d.y = uint32_t(regs.resolution[1] + 15) / 16;
			d.z = 1;

			if (final_output)
			{
				component_regs.output_offset = plane_offset[c];
				component_regs.output_stride = plane_stride[c];
				component_regs.output_rows = uint32_t(aligned_plane_height(c));
				write_uniform(uniform_data, slot, component_regs);

				const BindingResource resources[] = {
					bind_buffer(0, uniform_buffer, uint64_t(slot) * UniformSlotSize, sizeof(IDWTRegisters)),
					bind_view(1, pyramid.component_layer_views[c][input_level]),
					bind_sampler(2, device->mirror_repeat_sampler),
					bind_buffer(4, output_buffer),
				};
				d.pipeline = device->idwt_final.pipeline;
				d.bind_group = create_bind_group(device, device->idwt_final, resources, 4);
			}
			else
			{
				component_regs.store_fp16 = device->precision == 1 && (input_level - 1) < WaveletFP16Levels;
				write_uniform(uniform_data, slot, component_regs);

				const BindingResource resources[] = {
					bind_buffer(0, uniform_buffer, uint64_t(slot) * UniformSlotSize, sizeof(IDWTRegisters)),
					bind_view(1, pyramid.component_layer_views[c][input_level]),
					bind_sampler(2, device->mirror_repeat_sampler),
					bind_view(3, pyramid.component_ll_views[c][input_level - 1]),
				};
				d.pipeline = device->idwt.pipeline;
				d.bind_group = create_bind_group(device, device->idwt, resources, 4);
			}

			if (!d.bind_group)
				return false;
			dispatches.push_back(d);
			slot++;
		}
	}

	return true;
}

bool pyrowave_webgpu_decoder_opaque::init(pyrowave_webgpu_device device_, int width, int height, ChromaSubsampling chroma)
{
	device = device_;

	if (!layout.init(width, height, chroma))
		return false;
	parser.init(&layout);

	if (!pyramid.init(device, layout))
		return false;

	dequant_offset_buffer = create_buffer(device, uint64_t(layout.block_count_32x32) * sizeof(uint32_t),
	                                      WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "dequant-offsets");
	if (!dequant_offset_buffer)
		return false;

	payload_buffer_size = 64 * 1024;
	payload_buffer = create_buffer(device, payload_buffer_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
	                               "payload-data");
	if (!payload_buffer)
		return false;

	uint32_t offset = 0;
	for (int plane = 0; plane < NumComponents; plane++)
	{
		plane_offset[plane] = offset;
		plane_stride[plane] = uint32_t(aligned_plane_width(plane)) / 4;
		offset += plane_stride[plane] * uint32_t(aligned_plane_height(plane));
	}
	output_size = uint64_t(offset) * sizeof(uint32_t);
	output_buffer = create_buffer(device, output_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "decoded-planes");
	if (!output_buffer)
		return false;

	// At most 15 iDWT dispatches.
	constexpr uint32_t MaxUniformSlots = 15;
	uniform_buffer = create_buffer(device, uint64_t(MaxUniformSlots) * UniformSlotSize,
	                               WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "decoder-uniforms");
	if (!uniform_buffer)
		return false;

	uint32_t slot = 0;
	if (!plan_dequant() || !plan_idwt(slot) || !rebuild_dequant_groups())
		return false;
	if (slot > MaxUniformSlots)
		return false;

	// Nothing in here changes per frame.
	wgpuQueueWriteBuffer(device->queue, uniform_buffer, 0, uniform_data.data(), uniform_data.size());

	if (!timer.init(device, STAGE_COUNT))
		return false;

	readback_timestamp_offset = (output_size + 255) & ~uint64_t(255);
	return readback.ensure_size(device, readback_timestamp_offset + timer.readback_size(), "decoder-readback");
}

void pyrowave_webgpu_decoder_opaque::release()
{
	if (readback.pending)
		readback.wait(device);
	readback.release();
	timer.release();
	for (auto &table : dequant_bands)
		table.release();

	for (auto &stage : stages)
		for (auto &d : stage)
			if (d.bind_group)
				wgpuBindGroupRelease(d.bind_group);

	WGPUBuffer buffers[] = { dequant_offset_buffer, payload_buffer, output_buffer, uniform_buffer };
	for (auto buf : buffers)
		if (buf)
			wgpuBufferRelease(buf);

	pyramid.release();
}

static ChromaSubsampling to_chroma(pyrowave_webgpu_chroma_subsampling chroma)
{
	return chroma == PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_444 ? ChromaSubsampling::Chroma444 : ChromaSubsampling::Chroma420;
}

pyrowave_webgpu_result pyrowave_webgpu_decoder_create(const pyrowave_webgpu_decoder_create_info *info,
                                                      pyrowave_webgpu_decoder *out_decoder)
{
	if (!info || !out_decoder || !info->device)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	if (info->chroma != PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_420 && info->chroma != PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_444)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	if (info->width <= 0 || info->height <= 0 || info->width > 16384 || info->height > 16384)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	if (info->chroma == PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_420 && ((info->width | info->height) & 1))
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	auto *decoder = new (std::nothrow) pyrowave_webgpu_decoder_opaque;
	if (!decoder)
		return PYROWAVE_WEBGPU_ERROR_OUT_OF_HOST_MEMORY;

	if (!decoder->init(info->device, info->width, info->height, to_chroma(info->chroma)))
	{
		decoder->release();
		delete decoder;
		return PYROWAVE_WEBGPU_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	*out_decoder = decoder;
	return PYROWAVE_WEBGPU_SUCCESS;
}

void pyrowave_webgpu_decoder_destroy(pyrowave_webgpu_decoder decoder)
{
	if (!decoder)
		return;
	decoder->release();
	delete decoder;
}

void pyrowave_webgpu_decoder_clear(pyrowave_webgpu_decoder decoder)
{
	if (decoder)
		decoder->parser.clear();
}

pyrowave_webgpu_result pyrowave_webgpu_decoder_push_packet(pyrowave_webgpu_decoder decoder, const void *data, size_t size)
{
	if (!decoder || (!data && size))
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	return decoder->parser.push_packet(data, size) ? PYROWAVE_WEBGPU_SUCCESS : PYROWAVE_WEBGPU_ERROR_CORRUPT_BITSTREAM;
}

bool pyrowave_webgpu_decoder_decode_is_ready(pyrowave_webgpu_decoder decoder, bool allow_partial_frame)
{
	return decoder && decoder->parser.decode_is_ready(allow_partial_frame);
}

bool pyrowave_webgpu_decoder_decode_is_ready_with_sideband(pyrowave_webgpu_decoder decoder, bool allow_partial_frame,
                                                           int num_pristine_bands, float minimum_packet_ratio,
                                                           const uint32_t *active_block_mask, size_t word_count)
{
	if (!decoder || num_pristine_bands < 0 || num_pristine_bands >= DecompositionLevels)
		return false;
	return decoder->parser.decode_is_ready(allow_partial_frame, num_pristine_bands, minimum_packet_ratio,
	                                       active_block_mask, word_count);
}

pyrowave_webgpu_result pyrowave_webgpu_decoder_decode_submit(pyrowave_webgpu_decoder decoder)
{
	if (!decoder)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	auto *device = decoder->device;

	// The readback buffer is reused; the previous frame must be off the CPU first.
	if (decoder->readback.pending)
		decoder->readback.wait(device);
	decoder->readback.unmap();
	decoder->has_frame = false;

	const auto &offsets = decoder->parser.dequant_offsets();
	const auto &payload = decoder->parser.payload();

	// The dequant shader can read slightly past the end of the payload.
	uint64_t required = uint64_t(payload.size()) * sizeof(uint32_t) + 16;
	if (required > decoder->payload_buffer_size)
	{
		wgpuBufferRelease(decoder->payload_buffer);
		decoder->payload_buffer_size = std::max<uint64_t>(64 * 1024, required * 2);
		decoder->payload_buffer = create_buffer(device, decoder->payload_buffer_size,
		                                        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, "payload-data");
		if (!decoder->payload_buffer || !decoder->rebuild_dequant_groups())
			return PYROWAVE_WEBGPU_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	wgpuQueueWriteBuffer(device->queue, decoder->dequant_offset_buffer, 0, offsets.data(), offsets.size() * sizeof(uint32_t));
	if (!payload.empty())
		wgpuQueueWriteBuffer(device->queue, decoder->payload_buffer, 0, payload.data(), payload.size() * sizeof(uint32_t));

	WGPUCommandEncoder cmd = wgpuDeviceCreateCommandEncoder(device->device, nullptr);

	static const char *const stage_labels[STAGE_COUNT] = { "dequant", "idwt" };
	for (int stage = 0; stage < STAGE_COUNT; stage++)
	{
		WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
		pass_desc.label = string_view(stage_labels[stage]);
		pass_desc.timestampWrites = decoder->timer.writes(stage);
		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd, &pass_desc);
		record_dispatches(pass, decoder->stages[stage]);
		wgpuComputePassEncoderEnd(pass);
		wgpuComputePassEncoderRelease(pass);
	}

	wgpuCommandEncoderCopyBufferToBuffer(cmd, decoder->output_buffer, 0, decoder->readback.buffer, 0, decoder->output_size);
	decoder->timer.resolve(cmd, decoder->readback.buffer, decoder->readback_timestamp_offset);

	WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd, nullptr);
	wgpuQueueSubmit(device->queue, 1, &cmd_buffer);
	wgpuCommandBufferRelease(cmd_buffer);
	wgpuCommandEncoderRelease(cmd);

	decoder->readback.map_async(device);
	if (const char *dump = getenv("PYROWAVE_WEBGPU_DUMP_PYRAMID"))
		decoder->pyramid.dump(device, decoder->layout, dump);
	decoder->parser.mark_frame_decoded();
	decoder->has_frame = true;
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_decoder_decode_poll(pyrowave_webgpu_decoder decoder, bool wait)
{
	if (!decoder || !decoder->has_frame)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	if (wait)
		decoder->readback.wait(decoder->device);
	else if (decoder->readback.pending)
		decoder->device->process_events();

	if (decoder->readback.pending)
		return PYROWAVE_WEBGPU_NOT_READY;
	return decoder->readback.mapped ? PYROWAVE_WEBGPU_SUCCESS : PYROWAVE_WEBGPU_ERROR_GENERIC;
}

pyrowave_webgpu_result pyrowave_webgpu_decoder_decode_read(pyrowave_webgpu_decoder decoder,
                                                           const pyrowave_webgpu_cpu_buffer *output)
{
	if (!decoder || !output || !decoder->has_frame)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	const auto &layout = decoder->layout;
	if (output->width != layout.width || output->height != layout.height)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	auto expected = layout.chroma == ChromaSubsampling::Chroma420 ?
	                PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV420P : PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV444P;
	if (output->format != expected)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	for (int plane = 0; plane < NumComponents; plane++)
	{
		size_t row_bytes = size_t(decoder->plane_width(plane));
		size_t rows = size_t(decoder->plane_height(plane));
		if (!output->data[plane] || output->row_stride_in_bytes[plane] < row_bytes ||
		    output->plane_size_in_bytes[plane] < output->row_stride_in_bytes[plane] * (rows - 1) + row_bytes)
			return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	}

	if (!decoder->readback.wait(decoder->device))
		return PYROWAVE_WEBGPU_ERROR_GENERIC;

	auto *mapped = static_cast<const uint8_t *>(decoder->readback.data);
	for (int plane = 0; plane < NumComponents; plane++)
	{
		const uint8_t *src = mapped + size_t(decoder->plane_offset[plane]) * sizeof(uint32_t);
		size_t src_stride = size_t(decoder->plane_stride[plane]) * sizeof(uint32_t);
		auto *dst = static_cast<uint8_t *>(output->data[plane]);
		int w = decoder->plane_width(plane);
		int h = decoder->plane_height(plane);
		for (int y = 0; y < h; y++)
			memcpy(dst + size_t(y) * output->row_stride_in_bytes[plane], src + size_t(y) * src_stride, size_t(w));
	}

	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_decoder_decode_cpu(pyrowave_webgpu_decoder decoder,
                                                          const pyrowave_webgpu_cpu_buffer *output)
{
	auto result = pyrowave_webgpu_decoder_decode_submit(decoder);
	if (result != PYROWAVE_WEBGPU_SUCCESS)
		return result;
	return pyrowave_webgpu_decoder_decode_read(decoder, output);
}

pyrowave_webgpu_result pyrowave_webgpu_decoder_get_timings(pyrowave_webgpu_decoder decoder,
                                                           pyrowave_webgpu_timings *timings)
{
	if (!decoder || !timings || !decoder->has_frame || !decoder->readback.wait(decoder->device))
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	decoder->timer.read(static_cast<const uint8_t *>(decoder->readback.data) + decoder->readback_timestamp_offset,
	                    timings);
	return PYROWAVE_WEBGPU_SUCCESS;
}
