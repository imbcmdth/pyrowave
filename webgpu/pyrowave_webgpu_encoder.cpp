// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT

// WebGPU backend for the PyroWave encoder. Mirrors the compute pipeline of
// pyrowave_encoder.cpp: dwt -> quant -> analyze_rdo -> resolve_rdo -> block_packing
// on the GPU, then packetize on the CPU with the shared bitstream code.
//
// Every dispatch and its bind group is planned once at creation. A frame only
// rewrites the uniform data, records the passes and reads the result back.

#include "pyrowave_webgpu_common.hpp"
#include <algorithm>
#include <stdlib.h>
#include <cmath>
#include <new>

using namespace PyroWave;
using namespace PyroWave::WebGPU;

namespace
{
// Uniform layouts. These match the Registers structs in webgpu/shaders, which keep
// the field order of the GLSL push constant blocks.
struct DWTRegisters
{
	int32_t resolution[2];
	float inv_resolution[2];
	int32_t aligned_resolution[2];
	uint32_t store_fp16;
	uint32_t padding;
};
static_assert(sizeof(DWTRegisters) == 32, "Must match Registers in the WGSL.");

struct QuantizerRegisters
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	float inv_resolution[2];
	float input_layer;
	float quant_resolution;
	int32_t block_offset;
	int32_t block_stride;
	float rdo_distortion_scale;
	// The quantizer reads the whole pyramid, see plan_quant_analyze().
	int32_t texture_layer;
	int32_t texture_level;
	uint32_t padding[3];
};
static_assert(sizeof(QuantizerRegisters) == 64, "Must match Registers in the WGSL.");

struct AnalyzeRegisters
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	uint32_t total_wg_count;
	uint32_t num_blocks_aligned;
	uint32_t block_index_shamt;
	uint32_t padding;
};
static_assert(sizeof(AnalyzeRegisters) == 48, "Must match Registers in the WGSL.");

struct ResolveRegisters
{
	uint32_t target_payload_size;
	uint32_t num_blocks_per_subdivision;
	uint32_t padding[2];
};
static_assert(sizeof(ResolveRegisters) == 16, "Must match Registers in the WGSL.");

struct BlockPackingRegisters
{
	int32_t resolution[2];
	int32_t resolution_32x32_blocks[2];
	int32_t resolution_8x8_blocks[2];
	uint32_t quant_resolution_code;
	uint32_t sequence_code;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
	uint32_t padding[4];
};
static_assert(sizeof(BlockPackingRegisters) == 64, "Must match Registers in the WGSL.");

enum Stage
{
	STAGE_DWT,
	STAGE_QUANT,
	STAGE_ANALYZE,
	STAGE_RESOLVE,
	STAGE_PACKING,
	STAGE_COUNT
};

//////
// Initial quantization resolution and RDO weights. Lifted verbatim from
// Encoder::Impl in pyrowave_encoder.cpp, which cannot be shared as is because it
// lives in the Granite build; the only change is that the precision and chroma mode
// are passed in. Keep in sync.

float get_noise_power_normalized_quant_resolution(int precision, int level, int component, int band)
{
	// The initial quantization resolution aims for a flat spectrum with noise power normalization.
	// The low-pass gain for CDF 9/7 is 6 dB (1 bit). Every decomposition level subtracts 6 dB.

	// Maybe make this based on the max rate to have a decent initial estimate.
	int bits = precision >= 1 ? 8 : 6;

	if (band == 0)
		bits += 2;
	else if (band < 3)
		bits += 1;

	bits += level;

	// Chroma starts at level 1, subtract one bit.
	if (component != 0)
		bits--;

	return float(1 << bits);
}

float get_quant_resolution(int precision, int level, int component, int band)
{
	// FP16 range is limited, and this is more than a good enough initial estimate.
	return std::min<float>(
			precision >= 1 ? 4096.0f : 512.0f,
			get_noise_power_normalized_quant_resolution(precision, level, component, band));
}

float get_quant_rdo_distortion_scale(ChromaSubsampling chroma, int precision, int level, int component, int band)
{
	// From my Linelet master thesis. Copy paste 11 years later, ah yes :D
	float horiz_midpoint = (band & 1) ? 0.75f : 0.25f;
	float vert_midpoint = (band & 2) ? 0.75f : 0.25f;

	// Normal PC monitors.
	constexpr float dpi = 96.0f;
	// Compromise between couch gaming and desktop.
	constexpr float viewing_distance = 1.0f;
	constexpr float cpd_nyquist = 0.34f * viewing_distance * dpi;

	float cpd = std::sqrt(horiz_midpoint * horiz_midpoint + vert_midpoint * vert_midpoint) *
	            cpd_nyquist * std::exp2(-float(level));

	// Don't allow a situation where we're quantizing LL band hard.
	cpd = std::max(cpd, 8.0f);

	float csf = 2.6f * (0.0192f + 0.114f * cpd) * std::exp(-std::pow(0.114f * cpd, 1.1f));

	// Heavily discount chroma quality.
	if (component != 0 && level != DecompositionLevels - 1)
	{
		// Consider chroma a little more important if we're not subsampling.
		if (chroma == ChromaSubsampling::Chroma420)
			csf *= 0.6f;
	}

	// Due to filtering, distortion in lower bands will result in more noise power.
	// By scaling the distortion by this factor, we ensure uniform results.
	float resolution = get_noise_power_normalized_quant_resolution(precision, level, component, band);
	float weighted_resolution = csf * resolution;

	// The distortion is scaled in terms of power, not amplitude.
	return weighted_resolution * weighted_resolution;
}
//////

int floor_log2(uint32_t v)
{
	int result = -1;
	while (v)
	{
		v >>= 1;
		result++;
	}
	return result;
}

// A DWT dispatch that reads one of the input planes rather than the pyramid. Its bind
// group depends on the input, so it is rebuilt for GPU input.
struct InputDispatch
{
	size_t dispatch_index;
	int plane;
	uint32_t slot;
	int dc_shift;
	WGPUTextureView output;
};
}

struct pyrowave_webgpu_encoder_opaque
{
	pyrowave_webgpu_device device = nullptr;
	BlockLayout layout;
	WaveletPyramid pyramid;

	WGPUBuffer block_stat_buffer = nullptr;
	WGPUBuffer meta_buffer = nullptr;
	WGPUBuffer payload_data = nullptr;
	WGPUBuffer quant_buffer = nullptr;
	WGPUBuffer bucket_buffer = nullptr;
	uint64_t payload_data_size = 0;
	uint64_t bucket_buffer_size = 0;
	uint64_t quant_buffer_size = 0;

	// Outputs of block packing. The bitstream buffer is sized from the rate target.
	WGPUBuffer bitstream_buffer = nullptr;
	WGPUBuffer bitstream_meta_buffer = nullptr;
	uint64_t bitstream_size = 0;
	uint64_t meta_size = 0;

	WGPUBuffer uniform_buffer = nullptr;
	std::vector<uint8_t> uniform_data;

	// Input planes for the CPU entry point.
	WGPUTexture input_textures[3] = {};
	WGPUTextureView input_views[3] = {};
	std::vector<uint8_t> deinterleave_scratch[2];

	std::vector<Dispatch> stages[STAGE_COUNT];
	std::vector<InputDispatch> input_dispatches;
	std::vector<WGPUBindGroup> cpu_input_groups;
	uint32_t resolve_slot = 0;

	// Quantize, analyze and packing each run every band in one dispatch.
	BandTable quant_bands;
	BandTable analyze_bands;
	BandTable packing_bands;

	StageTimer timer;
	Readback readback;
	uint64_t readback_bitstream_offset = 0;
	uint64_t readback_timestamp_offset = 0;

	uint32_t sequence_count = 0;
	bool has_frame = false;

	bool init(pyrowave_webgpu_device device, int width, int height, ChromaSubsampling chroma);
	void release();

	bool plan_dwt(uint32_t &slot);
	bool plan_quant_analyze();
	bool plan_resolve(uint32_t &slot);
	bool plan_packing();
	bool rebuild_packing_groups();
	WGPUBindGroup create_input_group(const InputDispatch &input, WGPUTextureView plane);

	int plane_width(int plane) const;
	int plane_height(int plane) const;

	pyrowave_webgpu_result prepare_frame(const pyrowave_webgpu_rate_control *rate_control);
	pyrowave_webgpu_result submit_frame(const WGPUTextureView *planes);
	bool wait_result();
};

int pyrowave_webgpu_encoder_opaque::plane_width(int plane) const
{
	return plane != 0 && layout.chroma == ChromaSubsampling::Chroma420 ? layout.width / 2 : layout.width;
}

int pyrowave_webgpu_encoder_opaque::plane_height(int plane) const
{
	return plane != 0 && layout.chroma == ChromaSubsampling::Chroma420 ? layout.height / 2 : layout.height;
}

WGPUBindGroup pyrowave_webgpu_encoder_opaque::create_input_group(const InputDispatch &input, WGPUTextureView plane)
{
	const BindingResource resources[] = {
		bind_buffer(0, uniform_buffer, uint64_t(input.slot) * UniformSlotSize, sizeof(DWTRegisters)),
		bind_view(1, plane),
		bind_sampler(2, device->mirror_repeat_sampler),
		bind_view(3, input.output),
	};
	return create_bind_group(device, device->dwt[input.dc_shift], resources, 4);
}

bool pyrowave_webgpu_encoder_opaque::plan_dwt(uint32_t &slot)
{
	// Mirrors Encoder::Impl::dwt().
	auto &dispatches = stages[STAGE_DWT];
	const bool is_420 = layout.chroma == ChromaSubsampling::Chroma420;

	const auto add = [&](int output_level, int component, const DWTRegisters &regs, int dc_shift,
	                     int input_plane, WGPUTextureView input_ll) -> bool
	{
		write_uniform(uniform_data, slot, regs);
		WGPUTextureView output = pyramid.component_layer_views[component][output_level];
		Dispatch d = {};
		d.pipeline = device->dwt[dc_shift].pipeline;
		d.x = uint32_t(regs.aligned_resolution[0] + 31) / 32;
		d.y = uint32_t(regs.aligned_resolution[1] + 31) / 32;
		d.z = 1;

		if (input_plane >= 0)
		{
			input_dispatches.push_back({ dispatches.size(), input_plane, slot, dc_shift, output });
		}
		else
		{
			const BindingResource resources[] = {
				bind_buffer(0, uniform_buffer, uint64_t(slot) * UniformSlotSize, sizeof(DWTRegisters)),
				bind_view(1, input_ll),
				bind_sampler(2, device->mirror_repeat_sampler),
				bind_view(3, output),
			};
			d.bind_group = create_bind_group(device, device->dwt[dc_shift], resources, 4);
			if (!d.bind_group)
				return false;
		}

		dispatches.push_back(d);
		slot++;
		return true;
	};

	for (int output_level = 0; output_level < DecompositionLevels; output_level++)
	{
		DWTRegisters regs = {};
		regs.store_fp16 = device->precision == 1 && output_level < WaveletFP16Levels;

		if (output_level > 0)
		{
			regs.resolution[0] = layout.level_width(output_level - 1);
			regs.resolution[1] = layout.level_height(output_level - 1);
			regs.aligned_resolution[0] = regs.resolution[0];
			regs.aligned_resolution[1] = regs.resolution[1];
		}
		else
		{
			regs.resolution[0] = plane_width(0);
			regs.resolution[1] = plane_height(0);
			regs.aligned_resolution[0] = layout.aligned_width;
			regs.aligned_resolution[1] = layout.aligned_height;
		}

		regs.inv_resolution[0] = 1.0f / float(regs.resolution[0]);
		regs.inv_resolution[1] = 1.0f / float(regs.resolution[1]);

		if (output_level == 0)
		{
			int components = is_420 ? 1 : NumComponents;
			for (int c = 0; c < components; c++)
				if (!add(output_level, c, regs, 1, c, nullptr))
					return false;
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
				if (is_420 && c != 0 && output_level == 1)
				{
					// 420 chroma enters the pyramid at level 1, straight from the input plane.
					DWTRegisters chroma_regs = regs;
					chroma_regs.resolution[0] = plane_width(c);
					chroma_regs.resolution[1] = plane_height(c);
					chroma_regs.aligned_resolution[0] = layout.aligned_width >> output_level;
					chroma_regs.aligned_resolution[1] = layout.aligned_height >> output_level;
					chroma_regs.inv_resolution[0] = 1.0f / float(chroma_regs.resolution[0]);
					chroma_regs.inv_resolution[1] = 1.0f / float(chroma_regs.resolution[1]);
					if (!add(output_level, c, chroma_regs, 1, c, nullptr))
						return false;
				}
				else
				{
					if (!add(output_level, c, regs, 0, -1, pyramid.component_ll_views[c][output_level - 1]))
						return false;
				}
			}
		}
	}

	return true;
}

bool pyrowave_webgpu_encoder_opaque::plan_quant_analyze()
{
	// Mirrors Encoder::Impl::quant() and analyze_rdo(), except that every band goes
	// into one dispatch per stage, see shaders/band_dispatch.wgsl.
	const int per_subdivision = compute_block_count_per_subdivision(layout.block_count_32x32);

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				const int level_width = layout.level_width(level);
				const int level_height = layout.level_height(level);
				const auto &meta = layout.block_meta[component][level][band];

				float quant_res = get_quant_resolution(device->precision, level, component, band);

				QuantizerRegisters quant = {};
				quant.resolution[0] = level_width;
				quant.resolution[1] = level_height;
				quant.resolution_8x8_blocks[0] = (level_width + 7) / 8;
				quant.resolution_8x8_blocks[1] = (level_height + 7) / 8;
				quant.inv_resolution[0] = 1.0f / float(level_width);
				quant.inv_resolution[1] = 1.0f / float(level_height);
				quant.input_layer = float(band);
				quant.quant_resolution = 1.0f / decode_quant(encode_quant(1.0f / quant_res));
				quant.rdo_distortion_scale =
						get_quant_rdo_distortion_scale(layout.chroma, device->precision, level, component, band) *
						(1.0f / 256.0f);
				quant.block_offset = meta.block_offset_8x8;
				quant.block_stride = meta.block_stride_8x8;
				quant.texture_layer = NumFrequencyBandsPerLevel * component + band;
				quant.texture_level = level;
				quant_bands.add(quant, uint32_t(level_width + 31) / 32, uint32_t(level_height + 31) / 32);

				AnalyzeRegisters analyze = {};
				analyze.resolution[0] = level_width;
				analyze.resolution[1] = level_height;
				analyze.resolution_8x8_blocks[0] = (level_width + 7) / 8;
				analyze.resolution_8x8_blocks[1] = (level_height + 7) / 8;
				analyze.block_offset_8x8 = meta.block_offset_8x8;
				analyze.block_stride_8x8 = meta.block_stride_8x8;
				analyze.block_offset_32x32 = meta.block_offset_32x32;
				analyze.block_stride_32x32 = meta.block_stride_32x32;
				analyze.total_wg_count = uint32_t(layout.block_count_32x32);
				analyze.num_blocks_aligned = uint32_t(per_subdivision * BlockSpaceSubdivision);
				analyze.block_index_shamt = uint32_t(floor_log2(uint32_t(per_subdivision)));
				analyze_bands.add(analyze, uint32_t(level_width + 31) / 32, uint32_t(level_height + 31) / 32);
			}
		}
	}

	if (!quant_bands.create(device, "quant-bands") || !analyze_bands.create(device, "analyze-bands"))
		return false;

	{
		const BindingResource resources[] = {
			quant_bands.bind_registers(),
			bind_view(1, pyramid.full_view),
			bind_buffer(2, meta_buffer),
			bind_buffer(3, block_stat_buffer),
			bind_buffer(4, payload_data),
			quant_bands.bind_ranges(),
		};
		WGPUBindGroup group = create_bind_group(device, device->quant, resources, 6);
		if (!group)
			return false;
		stages[STAGE_QUANT].push_back(quant_bands.dispatch(device->quant.pipeline, group));
	}

	{
		const BindingResource resources[] = {
			analyze_bands.bind_registers(),
			bind_buffer(1, bucket_buffer),
			bind_buffer(2, block_stat_buffer),
			analyze_bands.bind_ranges(),
		};
		WGPUBindGroup group = create_bind_group(device, device->analyze, resources, 4);
		if (!group)
			return false;
		stages[STAGE_ANALYZE].push_back(analyze_bands.dispatch(device->analyze.pipeline, group));
	}

	const BindingResource resources[] = { bind_buffer(0, bucket_buffer) };
	Dispatch d = {};
	d.pipeline = device->analyze_finalize.pipeline;
	d.bind_group = create_bind_group(device, device->analyze_finalize, resources, 1);
	d.x = d.y = d.z = 1;
	if (!d.bind_group)
		return false;
	stages[STAGE_ANALYZE].push_back(d);
	return true;
}

bool pyrowave_webgpu_encoder_opaque::plan_resolve(uint32_t &slot)
{
	// Mirrors Encoder::Impl::resolve_rdo(). target_payload_size is filled in per frame.
	resolve_slot = slot++;
	ResolveRegisters regs = {};
	regs.num_blocks_per_subdivision = uint32_t(compute_block_count_per_subdivision(layout.block_count_32x32));
	write_uniform(uniform_data, resolve_slot, regs);

	const BindingResource resources[] = {
		bind_buffer(0, uniform_buffer, uint64_t(resolve_slot) * UniformSlotSize, sizeof(ResolveRegisters)),
		bind_buffer(1, bucket_buffer),
		bind_buffer(2, quant_buffer),
	};
	Dispatch d = {};
	d.pipeline = device->resolve.pipeline;
	d.bind_group = create_bind_group(device, device->resolve, resources, 3);
	d.x = NumRDOBuckets * BlockSpaceSubdivision;
	d.y = d.z = 1;
	if (!d.bind_group)
		return false;
	stages[STAGE_RESOLVE].push_back(d);
	return true;
}

bool pyrowave_webgpu_encoder_opaque::plan_packing()
{
	// Mirrors Encoder::Impl::block_packing(), one dispatch for all bands. The bind
	// group references the bitstream buffer, so it is created by
	// rebuild_packing_groups(), and the sequence code is filled in per frame.
	for (int level = 0; level < DecompositionLevels; level++)
	{
		const int level_width = layout.level_width(level);
		const int level_height = layout.level_height(level);

		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				BlockPackingRegisters regs = {};
				regs.resolution[0] = level_width;
				regs.resolution[1] = level_height;
				regs.resolution_32x32_blocks[0] = (level_width + 31) / 32;
				regs.resolution_32x32_blocks[1] = (level_height + 31) / 32;
				regs.resolution_8x8_blocks[0] = (level_width + 7) / 8;
				regs.resolution_8x8_blocks[1] = (level_height + 7) / 8;

				auto quant_res = get_quant_resolution(device->precision, level, component, band);
				regs.quant_resolution_code = encode_quant(1.0f / quant_res);

				const auto &meta = layout.block_meta[component][level][band];
				regs.block_offset_32x32 = meta.block_offset_32x32;
				regs.block_stride_32x32 = meta.block_stride_32x32;
				regs.block_offset_8x8 = meta.block_offset_8x8;
				regs.block_stride_8x8 = meta.block_stride_8x8;

				// A workgroup covers 2x2 of the 32x32 blocks.
				packing_bands.add(regs, uint32_t(regs.resolution_32x32_blocks[0] + 1) / 2,
				                  uint32_t(regs.resolution_32x32_blocks[1] + 1) / 2);
			}
		}
	}

	if (!packing_bands.create(device, "packing-bands"))
		return false;
	stages[STAGE_PACKING].push_back(packing_bands.dispatch(device->block_packing.pipeline, nullptr));
	return true;
}

bool pyrowave_webgpu_encoder_opaque::rebuild_packing_groups()
{
	auto &d = stages[STAGE_PACKING].front();
	if (d.bind_group)
		wgpuBindGroupRelease(d.bind_group);

	const BindingResource resources[] = {
		packing_bands.bind_registers(),
		bind_buffer(1, bitstream_buffer),
		bind_buffer(2, bitstream_meta_buffer),
		bind_buffer(3, meta_buffer),
		bind_buffer(4, payload_data),
		bind_buffer(5, block_stat_buffer),
		bind_buffer(6, quant_buffer),
		packing_bands.bind_ranges(),
	};
	d.bind_group = create_bind_group(device, device->block_packing, resources, 8);
	return d.bind_group != nullptr;
}

bool pyrowave_webgpu_encoder_opaque::init(pyrowave_webgpu_device device_, int width, int height, ChromaSubsampling chroma)
{
	device = device_;

	if (!layout.init(width, height, chroma))
		return false;

	if (!pyramid.init(device, layout))
		return false;

	// Same sizes as Encoder::Impl::init_block_meta().
	const WGPUBufferUsage storage = WGPUBufferUsage_Storage;
	const WGPUBufferUsage storage_clear = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

	block_stat_buffer = create_buffer(device, uint64_t(layout.block_count_8x8) * sizeof(BlockStatsBlock),
	                                  storage, "block-stat-buffer");
	meta_buffer = create_buffer(device, uint64_t(layout.block_count_8x8) * sizeof(BlockMeta), storage, "meta-buffer");

	// Worst case estimate.
	payload_data_size = uint64_t(layout.aligned_width) * layout.aligned_height * 2;
	payload_data = create_buffer(device, payload_data_size, storage_clear, "payload-data");

	quant_buffer_size = (uint64_t(layout.block_count_32x32) * sizeof(uint32_t) + 3) & ~uint64_t(3);
	quant_buffer = create_buffer(device, quant_buffer_size, storage_clear, "quant-buffer");

	bucket_buffer_size = RDOBucketOffset;
	bucket_buffer_size += NumRDOBuckets * BlockSpaceSubdivision * sizeof(uint32_t);
	bucket_buffer_size += uint64_t(NumRDOBuckets) * compute_block_count_per_subdivision(layout.block_count_32x32) *
	                      BlockSpaceSubdivision * sizeof(RDOperation);
	bucket_buffer = create_buffer(device, bucket_buffer_size, storage_clear, "bucket-buffer");

	meta_size = uint64_t(layout.block_count_32x32) * sizeof(BitstreamPacket);
	bitstream_meta_buffer = create_buffer(device, meta_size, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc,
	                                      "bitstream-meta");

	if (!block_stat_buffer || !meta_buffer || !payload_data || !quant_buffer || !bucket_buffer || !bitstream_meta_buffer)
		return false;

	for (int plane = 0; plane < NumComponents; plane++)
	{
		WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
		desc.label = string_view(plane == 0 ? "input-y" : (plane == 1 ? "input-cb" : "input-cr"));
		desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
		desc.dimension = WGPUTextureDimension_2D;
		desc.size.width = uint32_t(plane_width(plane));
		desc.size.height = uint32_t(plane_height(plane));
		desc.size.depthOrArrayLayers = 1;
		desc.format = WGPUTextureFormat_R8Unorm;
		desc.mipLevelCount = 1;
		desc.sampleCount = 1;
		input_textures[plane] = wgpuDeviceCreateTexture(device->device, &desc);
		if (!input_textures[plane])
			return false;
		input_views[plane] = wgpuTextureCreateView(input_textures[plane], nullptr);
		if (!input_views[plane])
			return false;
	}

	// One uniform slot per dispatch that has its own: at most 15 DWT dispatches (444)
	// and resolve. The batched stages keep theirs in their band tables.
	constexpr uint32_t MaxUniformSlots = 15 + 1;
	uniform_buffer = create_buffer(device, uint64_t(MaxUniformSlots) * UniformSlotSize,
	                               WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, "encoder-uniforms");
	if (!uniform_buffer)
		return false;

	uint32_t slot = 0;
	if (!plan_dwt(slot) || !plan_quant_analyze() || !plan_resolve(slot) || !plan_packing())
		return false;
	if (slot > MaxUniformSlots)
		return false;

	for (auto &input : input_dispatches)
	{
		WGPUBindGroup group = create_input_group(input, input_views[input.plane]);
		if (!group)
			return false;
		cpu_input_groups.push_back(group);
	}

	return timer.init(device, STAGE_COUNT);
}

void pyrowave_webgpu_encoder_opaque::release()
{
	if (readback.pending)
		readback.wait(device);
	readback.release();
	timer.release();
	quant_bands.release();
	analyze_bands.release();
	packing_bands.release();

	for (auto &stage : stages)
		for (auto &d : stage)
			if (d.bind_group)
				wgpuBindGroupRelease(d.bind_group);
	for (auto group : cpu_input_groups)
		wgpuBindGroupRelease(group);

	for (auto &view : input_views)
		if (view)
			wgpuTextureViewRelease(view);
	for (auto &tex : input_textures)
		if (tex)
			wgpuTextureRelease(tex);

	WGPUBuffer buffers[] = {
		block_stat_buffer, meta_buffer, payload_data, quant_buffer, bucket_buffer,
		bitstream_buffer, bitstream_meta_buffer, uniform_buffer,
	};
	for (auto buf : buffers)
		if (buf)
			wgpuBufferRelease(buf);

	pyramid.release();
}

bool pyrowave_webgpu_encoder_opaque::wait_result()
{
	if (!has_frame)
		return false;
	return readback.wait(device);
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_opaque::prepare_frame(const pyrowave_webgpu_rate_control *rate_control)
{
	uint64_t target_size = rate_control->maximum_bitstream_size & ~uint64_t(3);

	// Check for bogus sizes.
	if (target_size > UINT32_MAX || target_size == 0)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	// The previous result must be off the CPU before its buffer can be written again.
	if (readback.pending)
		readback.wait(device);
	readback.unmap();
	has_frame = false;

	// Same slack the Vulkan C API gives the packer.
	uint64_t required_bitstream = target_size + meta_size;
	if (!bitstream_buffer || bitstream_size < required_bitstream)
	{
		if (bitstream_buffer)
			wgpuBufferRelease(bitstream_buffer);
		bitstream_size = required_bitstream;
		bitstream_buffer = create_buffer(device, bitstream_size,
		                                 WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, "bitstream");
		if (!bitstream_buffer || !rebuild_packing_groups())
			return PYROWAVE_WEBGPU_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	readback_bitstream_offset = (meta_size + 255) & ~uint64_t(255);
	readback_timestamp_offset = (readback_bitstream_offset + bitstream_size + 255) & ~uint64_t(255);
	if (!readback.ensure_size(device, readback_timestamp_offset + timer.readback_size(), "encoder-readback"))
		return PYROWAVE_WEBGPU_ERROR_OUT_OF_DEVICE_MEMORY;

	sequence_count = (sequence_count + 1) & SequenceCountMask;

	// The sequence header is part of the frame's budget.
	uint64_t target_payload_size = target_size;
	if (target_payload_size >= sizeof(BitstreamSequenceHeader))
		target_payload_size -= sizeof(BitstreamSequenceHeader);

	auto *resolve = reinterpret_cast<ResolveRegisters *>(uniform_data.data() + resolve_slot * size_t(UniformSlotSize));
	resolve->target_payload_size = uint32_t(target_payload_size / sizeof(uint32_t));
	for (uint32_t i = 0; i < packing_bands.count; i++)
		packing_bands.get<BlockPackingRegisters>(i).sequence_code = sequence_count;

	wgpuQueueWriteBuffer(device->queue, uniform_buffer, 0, uniform_data.data(), uniform_data.size());
	wgpuQueueWriteBuffer(device->queue, packing_bands.buffer, 0,
	                     packing_bands.registers.data(), packing_bands.registers.size());
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_opaque::submit_frame(const WGPUTextureView *planes)
{
	std::vector<WGPUBindGroup> temporary_groups;
	auto &dwt = stages[STAGE_DWT];

	for (size_t i = 0; i < input_dispatches.size(); i++)
	{
		auto &input = input_dispatches[i];
		if (planes)
		{
			WGPUBindGroup group = create_input_group(input, planes[input.plane]);
			if (!group)
			{
				for (auto g : temporary_groups)
					wgpuBindGroupRelease(g);
				return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
			}
			temporary_groups.push_back(group);
			dwt[input.dispatch_index].bind_group = group;
		}
		else
			dwt[input.dispatch_index].bind_group = cpu_input_groups[i];
	}

	WGPUCommandEncoder cmd = wgpuDeviceCreateCommandEncoder(device->device, nullptr);

	// The quantizer and packer OR their output into the payload scratch buffer, so all
	// of it has to start at zero. The GLSL only needs its two counters cleared.
	wgpuCommandEncoderClearBuffer(cmd, payload_data, 0, (payload_data_size + 3) & ~uint64_t(3));
	wgpuCommandEncoderClearBuffer(cmd, bucket_buffer, 0, (bucket_buffer_size + 3) & ~uint64_t(3));
	wgpuCommandEncoderClearBuffer(cmd, quant_buffer, 0, quant_buffer_size);

	static const char *const stage_labels[STAGE_COUNT] = { "dwt", "quant", "analyze", "resolve", "packing" };
	for (int stage = 0; stage < STAGE_COUNT; stage++)
	{
		WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
		pass_desc.label = string_view(stage_labels[stage]);
		pass_desc.timestampWrites = timer.writes(stage);
		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd, &pass_desc);
		record_dispatches(pass, stages[stage]);
		wgpuComputePassEncoderEnd(pass);
		wgpuComputePassEncoderRelease(pass);
	}

	wgpuCommandEncoderCopyBufferToBuffer(cmd, bitstream_meta_buffer, 0, readback.buffer, 0, (meta_size + 3) & ~uint64_t(3));
	wgpuCommandEncoderCopyBufferToBuffer(cmd, bitstream_buffer, 0, readback.buffer, readback_bitstream_offset,
	                                     (bitstream_size + 3) & ~uint64_t(3));
	timer.resolve(cmd, readback.buffer, readback_timestamp_offset);

	WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(cmd, nullptr);
	wgpuQueueSubmit(device->queue, 1, &cmd_buffer);
	wgpuCommandBufferRelease(cmd_buffer);
	wgpuCommandEncoderRelease(cmd);

	for (auto g : temporary_groups)
		wgpuBindGroupRelease(g);
	for (auto &input : input_dispatches)
		dwt[input.dispatch_index].bind_group = nullptr;

	readback.map_async(device);
	if (const char *dump = getenv("PYROWAVE_WEBGPU_DUMP_PYRAMID"))
		pyramid.dump(device, layout, dump);
	has_frame = true;
	return PYROWAVE_WEBGPU_SUCCESS;
}

static ChromaSubsampling to_chroma(pyrowave_webgpu_chroma_subsampling chroma)
{
	return chroma == PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_444 ? ChromaSubsampling::Chroma444 : ChromaSubsampling::Chroma420;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_create(const pyrowave_webgpu_encoder_create_info *info,
                                                      pyrowave_webgpu_encoder *out_encoder)
{
	if (!info || !out_encoder || !info->device)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	if (info->chroma != PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_420 && info->chroma != PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_444)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	if (info->width <= 0 || info->height <= 0 || info->width > 16384 || info->height > 16384)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	if (info->chroma == PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_420 && ((info->width | info->height) & 1))
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	if (!create_encode_pipelines(info->device))
		return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;

	auto *encoder = new (std::nothrow) pyrowave_webgpu_encoder_opaque;
	if (!encoder)
		return PYROWAVE_WEBGPU_ERROR_OUT_OF_HOST_MEMORY;

	if (!encoder->init(info->device, info->width, info->height, to_chroma(info->chroma)))
	{
		info->device->log("Failed to create encoder resources.\n");
		encoder->release();
		delete encoder;
		return PYROWAVE_WEBGPU_ERROR_OUT_OF_DEVICE_MEMORY;
	}

	*out_encoder = encoder;
	return PYROWAVE_WEBGPU_SUCCESS;
}

void pyrowave_webgpu_encoder_destroy(pyrowave_webgpu_encoder encoder)
{
	if (!encoder)
		return;
	encoder->release();
	delete encoder;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_encode_cpu(pyrowave_webgpu_encoder encoder,
                                                          const pyrowave_webgpu_cpu_buffer *input,
                                                          const pyrowave_webgpu_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	const auto &layout = encoder->layout;
	if (input->width != layout.width || input->height != layout.height)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	const bool is_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const bool nv12 = input->format == PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_NV12;
	if (nv12 || input->format == PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV420P)
	{
		if (!is_420)
			return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	}
	else if (input->format == PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV444P)
	{
		if (is_420)
			return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	}
	else
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	int num_planes = nv12 ? 2 : 3;
	for (int plane = 0; plane < num_planes; plane++)
	{
		size_t row_bytes = size_t(encoder->plane_width(plane)) * (nv12 && plane == 1 ? 2 : 1);
		size_t rows = size_t(encoder->plane_height(plane));
		if (!input->data[plane] || input->row_stride_in_bytes[plane] < row_bytes ||
		    input->plane_size_in_bytes[plane] < input->row_stride_in_bytes[plane] * (rows - 1) + row_bytes)
			return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	}

	auto result = encoder->prepare_frame(rate_control);
	if (result != PYROWAVE_WEBGPU_SUCCESS)
		return result;

	const auto upload = [&](int plane, const void *data, size_t stride, size_t size)
	{
		WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
		dst.texture = encoder->input_textures[plane];
		WGPUTexelCopyBufferLayout src_layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
		src_layout.bytesPerRow = uint32_t(stride);
		src_layout.rowsPerImage = uint32_t(encoder->plane_height(plane));
		WGPUExtent3D extent = { uint32_t(encoder->plane_width(plane)), uint32_t(encoder->plane_height(plane)), 1 };
		wgpuQueueWriteTexture(encoder->device->queue, &dst, data, size, &src_layout, &extent);
	};

	upload(0, input->data[0], input->row_stride_in_bytes[0], input->plane_size_in_bytes[0]);

	if (nv12)
	{
		// The DWT reads one channel per plane, so split the interleaved chroma here.
		int w = encoder->plane_width(1);
		int h = encoder->plane_height(1);
		for (auto &scratch : encoder->deinterleave_scratch)
			scratch.resize(size_t(w) * h);
		for (int y = 0; y < h; y++)
		{
			auto *src = static_cast<const uint8_t *>(input->data[1]) + y * input->row_stride_in_bytes[1];
			uint8_t *cb = encoder->deinterleave_scratch[0].data() + size_t(y) * w;
			uint8_t *cr = encoder->deinterleave_scratch[1].data() + size_t(y) * w;
			for (int x = 0; x < w; x++)
			{
				cb[x] = src[2 * x + 0];
				cr[x] = src[2 * x + 1];
			}
		}
		upload(1, encoder->deinterleave_scratch[0].data(), size_t(w), encoder->deinterleave_scratch[0].size());
		upload(2, encoder->deinterleave_scratch[1].data(), size_t(w), encoder->deinterleave_scratch[1].size());
	}
	else
	{
		upload(1, input->data[1], input->row_stride_in_bytes[1], input->plane_size_in_bytes[1]);
		upload(2, input->data[2], input->row_stride_in_bytes[2], input->plane_size_in_bytes[2]);
	}

	return encoder->submit_frame(nullptr);
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_encode_gpu(pyrowave_webgpu_encoder encoder,
                                                          const pyrowave_webgpu_gpu_input *input,
                                                          const pyrowave_webgpu_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	for (auto *plane : input->planes)
		if (!plane)
			return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	auto result = encoder->prepare_frame(rate_control);
	if (result != PYROWAVE_WEBGPU_SUCCESS)
		return result;
	return encoder->submit_frame(input->planes);
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_poll(pyrowave_webgpu_encoder encoder, bool wait)
{
	if (!encoder || !encoder->has_frame)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	if (wait)
		encoder->readback.wait(encoder->device);
	else if (encoder->readback.pending)
		encoder->device->process_events();

	if (encoder->readback.pending)
		return PYROWAVE_WEBGPU_NOT_READY;
	return encoder->readback.mapped ? PYROWAVE_WEBGPU_SUCCESS : PYROWAVE_WEBGPU_ERROR_GENERIC;
}

static const void *mapped_meta(pyrowave_webgpu_encoder encoder)
{
	return encoder->readback.data;
}

static const void *mapped_bitstream(pyrowave_webgpu_encoder encoder)
{
	return static_cast<const uint8_t *>(encoder->readback.data) + encoder->readback_bitstream_offset;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_compute_num_packets(pyrowave_webgpu_encoder encoder,
                                                                   size_t packet_boundary, size_t *num_packets)
{
	return pyrowave_webgpu_encoder_compute_num_packets_with_padding(encoder, packet_boundary, 0, num_packets);
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_compute_num_packets_with_padding(
		pyrowave_webgpu_encoder encoder, size_t packet_boundary, size_t padding_size, size_t *num_packets)
{
	if (!encoder || !num_packets || !encoder->wait_result())
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	*num_packets = compute_num_packets(encoder->layout, mapped_meta(encoder), packet_boundary, padding_size);
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_compute_num_critical_packets(
		pyrowave_webgpu_encoder encoder, int bands, size_t packet_boundary, size_t padding_size, size_t *num_packets)
{
	if (!encoder || !num_packets || bands >= DecompositionLevels || !encoder->wait_result())
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	*num_packets = compute_num_critical_packets(encoder->layout, bands, mapped_meta(encoder),
	                                            packet_boundary, padding_size);
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_packetize(pyrowave_webgpu_encoder encoder,
                                                         pyrowave_webgpu_packet *packets,
                                                         size_t packet_boundary, size_t *out_packets,
                                                         void *bitstream, size_t size)
{
	return pyrowave_webgpu_encoder_packetize_with_padding(encoder, packets, packet_boundary, 0,
	                                                      out_packets, bitstream, size);
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_packetize_with_padding(
		pyrowave_webgpu_encoder encoder, pyrowave_webgpu_packet *packets, size_t packet_boundary, size_t padding_size,
		size_t *out_packets, void *bitstream, size_t size)
{
	if (!encoder || !packets || !out_packets || !bitstream || !encoder->wait_result())
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	static_assert(sizeof(pyrowave_webgpu_packet) == sizeof(Packet), "Packet layout mismatch.");
	*out_packets = packetize(encoder->layout, reinterpret_cast<Packet *>(packets), packet_boundary,
	                         bitstream, size, mapped_meta(encoder), mapped_bitstream(encoder), padding_size);
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_get_mapped_raw_bitstream(
		pyrowave_webgpu_encoder encoder, const void **bitstream, size_t *bitstream_size,
		const void **metadata, size_t *metadata_size)
{
	if (!encoder || !bitstream || !bitstream_size || !metadata || !metadata_size || !encoder->wait_result())
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	*bitstream = mapped_bitstream(encoder);
	*bitstream_size = size_t(encoder->bitstream_size);
	*metadata = mapped_meta(encoder);
	*metadata_size = size_t(encoder->meta_size);
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_get_num_active_blocks(pyrowave_webgpu_encoder encoder, int bands,
                                                                     size_t *num_active_blocks)
{
	if (!encoder || !num_active_blocks || bands >= DecompositionLevels)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	*num_active_blocks = get_num_active_blocks(encoder->layout, bands);
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_compute_block_active_words(pyrowave_webgpu_encoder encoder,
                                                                          int bands, uint32_t *words, size_t word_count)
{
	if (!encoder || !words || bands >= DecompositionLevels || !encoder->wait_result())
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	if (word_count * 32 < get_num_active_blocks(encoder->layout, bands))
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	compute_block_active_words(encoder->layout, bands, words, word_count, mapped_meta(encoder));
	return PYROWAVE_WEBGPU_SUCCESS;
}

pyrowave_webgpu_result pyrowave_webgpu_encoder_get_timings(pyrowave_webgpu_encoder encoder,
                                                           pyrowave_webgpu_timings *timings)
{
	if (!encoder || !timings || !encoder->wait_result())
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;
	encoder->timer.read(static_cast<const uint8_t *>(encoder->readback.data) + encoder->readback_timestamp_offset,
	                    timings);
	return PYROWAVE_WEBGPU_SUCCESS;
}
