// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT
#pragma once

// Internal glue shared by the WebGPU encoder and decoder: the object behind
// pyrowave_webgpu_device, pipeline creation, waiting on futures, and the wavelet
// coefficient pyramid, which both directions allocate identically.

#include "pyrowave_webgpu.h"
#include "pyrowave_bitstream.hpp"

#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

namespace PyroWave
{
namespace WebGPU
{
// Every dispatch reads its parameters (the GLSL push constants) from its own slot of
// one uniform buffer. 256 is the largest minUniformBufferOffsetAlignment WebGPU allows.
constexpr uint32_t UniformSlotSize = 256;

// Same meaning and same env var as the Vulkan build. See pyrowave_webgpu.h.
constexpr int DefaultPrecision = 1;

// Levels below this are stored as FP16 under precision 1, as in pyrowave_common.cpp.
constexpr int WaveletFP16Levels = 2;

WGPUStringView string_view(const char *str);
std::string to_string(WGPUStringView view);

struct Pipeline
{
	WGPUComputePipeline pipeline = nullptr;
	WGPUBindGroupLayout layout = nullptr;
	void release();
};

enum class BindingType
{
	Uniform,
	StorageRead,
	StorageReadWrite,
	Texture2D,        // unfilterable float
	Texture2DArray,   // unfilterable float
	Sampler,          // non-filtering
	StorageTexture2D, // r32float, write only
	StorageTexture2DArray
};

struct BindingLayout
{
	uint32_t binding;
	BindingType type;
};

struct BindingResource
{
	uint32_t binding;
	WGPUBuffer buffer;
	uint64_t offset;
	uint64_t size;
	WGPUSampler sampler;
	WGPUTextureView view;
};

inline BindingResource bind_buffer(uint32_t binding, WGPUBuffer buffer, uint64_t offset = 0,
                                   uint64_t size = WGPU_WHOLE_SIZE)
{
	return { binding, buffer, offset, size, nullptr, nullptr };
}

inline BindingResource bind_sampler(uint32_t binding, WGPUSampler sampler)
{
	return { binding, nullptr, 0, 0, sampler, nullptr };
}

inline BindingResource bind_view(uint32_t binding, WGPUTextureView view)
{
	return { binding, nullptr, 0, 0, nullptr, view };
}

// A dispatch recorded once and replayed every frame.
struct Dispatch
{
	WGPUComputePipeline pipeline;
	WGPUBindGroup bind_group;
	uint32_t x, y, z;
};

// The wavelet coefficient pyramid. The decoder fills it from the bitstream and runs
// the iDWT out of it; the encoder runs the DWT into it and quantizes out of it.
//
// Always r32float: it is the only single channel float format WebGPU can write from
// a shader without optional features. Precision 1 rounds the two highest resolution
// levels through FP16 in the shaders instead, to match the Vulkan build's R16F.
struct WaveletPyramid
{
	WGPUTexture texture = nullptr;
	// 4 layer array views, one per component and level.
	WGPUTextureView component_layer_views[NumComponents][DecompositionLevels] = {};
	// Single layer 2D views of band 0 (LL).
	WGPUTextureView component_ll_views[NumComponents][DecompositionLevels] = {};
	// All 12 layers of one level, for dispatches that cover every band of a level.
	WGPUTextureView level_views[DecompositionLevels] = {};
	// Everything, for the quantizer, which reads all levels in one dispatch.
	WGPUTextureView full_view = nullptr;

	bool init(pyrowave_webgpu_device device, const BlockLayout &layout);
	void dump(pyrowave_webgpu_device device, const BlockLayout &layout, const char *path);
	void release();
};

// Every band of a stage in one dispatch; see shaders/band_dispatch.wgsl. Holds the
// per band register blocks (binding 0, array<Registers>) and the workgroup ranges
// (binding 7, array<vec4<u32>>) in one storage buffer.
struct BandTable
{
	WGPUBuffer buffer = nullptr;
	std::vector<uint8_t> registers;
	std::vector<uint32_t> ranges;
	uint32_t register_size = 0;
	uint32_t count = 0;
	uint32_t total_workgroups = 0;

	template <typename T>
	void add(const T &regs, uint32_t workgroups_x, uint32_t workgroups_y)
	{
		static_assert(sizeof(T) % 16 == 0, "Registers must be a multiple of 16 bytes.");
		register_size = sizeof(T);
		size_t offset = registers.size();
		registers.resize(offset + sizeof(T));
		memcpy(registers.data() + offset, &regs, sizeof(T));
		uint32_t workgroups = workgroups_x * workgroups_y;
		ranges.insert(ranges.end(), { total_workgroups, workgroups_x, workgroups, 0u });
		total_workgroups += workgroups;
		count++;
	}

	template <typename T>
	T &get(uint32_t index)
	{
		return *reinterpret_cast<T *>(registers.data() + size_t(index) * register_size);
	}

	uint64_t ranges_offset() const { return (uint64_t(registers.size()) + 255) & ~uint64_t(255); }
	bool create(pyrowave_webgpu_device device, const char *label);
	void upload(pyrowave_webgpu_device device);
	BindingResource bind_registers() const { return bind_buffer(0, buffer, 0, registers.size()); }
	BindingResource bind_ranges() const { return bind_buffer(7, buffer, ranges_offset(), ranges.size() * sizeof(uint32_t)); }
	Dispatch dispatch(WGPUComputePipeline pipeline, WGPUBindGroup group) const;
	void release();
};

// Timestamps around each stage's compute pass, if the device has timestamp queries.
struct StageTimer
{
	WGPUQuerySet query_set = nullptr;
	WGPUBuffer resolve_buffer = nullptr;
	int num_stages = 0;

	bool init(pyrowave_webgpu_device device, int stages);
	// NULL when timestamps are off.
	const WGPUPassTimestampWrites *writes(int stage);
	// Resolves into readback at readback_offset, which must be 8 byte aligned.
	void resolve(WGPUCommandEncoder cmd, WGPUBuffer readback, uint64_t readback_offset);
	uint64_t readback_size() const { return query_set ? uint64_t(num_stages) * 2 * sizeof(uint64_t) : 0; }
	void read(const void *mapped, pyrowave_webgpu_timings *timings) const;
	void release();

private:
	WGPUPassTimestampWrites stage_writes[8] = {};
};

// Readback of one buffer through wgpuBufferMapAsync.
struct Readback
{
	WGPUBuffer buffer = nullptr;
	uint64_t size = 0;
	bool pending = false;
	bool completed = false;
	bool mapped = false;
	bool failed = false;
	const void *data = nullptr;
	WGPUFuture future = {};

	bool ensure_size(pyrowave_webgpu_device device, uint64_t size, const char *label);
	// Call after the submit that fills it.
	void map_async(pyrowave_webgpu_device device);
	// Blocks until the map has completed. Returns false if it failed.
	bool wait(pyrowave_webgpu_device device);
	// Unmaps before the buffer is used again by the GPU.
	void unmap();
	void release();
};

// Encode pipeline creation is deferred to the first encoder.
bool create_encode_pipelines(pyrowave_webgpu_device device);

WGPUBuffer create_buffer(pyrowave_webgpu_device device, uint64_t size, WGPUBufferUsage usage, const char *label);
WGPUBindGroup create_bind_group(pyrowave_webgpu_device device, const Pipeline &pipeline,
                                const BindingResource *resources, size_t count);

template <typename T>
static inline void write_uniform(std::vector<uint8_t> &data, uint32_t slot, const T &value)
{
	static_assert(sizeof(T) <= UniformSlotSize, "Uniform data too large.");
	if (data.size() < (slot + 1) * size_t(UniformSlotSize))
		data.resize((slot + 1) * size_t(UniformSlotSize));
	memcpy(data.data() + slot * size_t(UniformSlotSize), &value, sizeof(T));
}

void record_dispatches(WGPUComputePassEncoder pass, const std::vector<Dispatch> &dispatches);
}
}

struct pyrowave_webgpu_device_opaque
{
	WGPUInstance instance = nullptr;
	WGPUAdapter adapter = nullptr;
	WGPUDevice device = nullptr;
	WGPUQueue queue = nullptr;

	uint32_t subgroup_min_size = 0;
	uint32_t subgroup_max_size = 0;
	bool timestamps = false;
	bool timed_wait_any = false;
	// False when subgroups came from wgpu-native's own feature, whose WGSL front end
	// rejects the standard `enable subgroups;` and allows the builtins without it.
	bool wgsl_enable_subgroups = true;
	bool device_lost = false;
	int precision = PyroWave::WebGPU::DefaultPrecision;
	uint32_t uniform_alignment = PyroWave::WebGPU::UniformSlotSize;

	WGPUSampler mirror_repeat_sampler = nullptr;

	// Decode, compiled by pyrowave_webgpu_device_create().
	PyroWave::WebGPU::Pipeline dequant;
	PyroWave::WebGPU::Pipeline idwt;       // Writes the next level's LL band.
	PyroWave::WebGPU::Pipeline idwt_final; // Writes 8-bit output planes.

	// Encode, compiled by the first pyrowave_webgpu_encoder_create().
	PyroWave::WebGPU::Pipeline dwt[2]; // Indexed by DCShift.
	PyroWave::WebGPU::Pipeline quant;
	PyroWave::WebGPU::Pipeline analyze;
	PyroWave::WebGPU::Pipeline analyze_finalize;
	PyroWave::WebGPU::Pipeline resolve;
	PyroWave::WebGPU::Pipeline block_packing;
	bool encode_pipelines_ready = false;

	pyrowave_webgpu_message_cb message_cb = nullptr;
	void *message_userdata = nullptr;

	void log(const char *fmt, ...) const
#if defined(__GNUC__)
	__attribute__((format(printf, 2, 3)))
#endif
	;

	// Blocks until *flag is set by a callback created with WGPUCallbackMode_AllowProcessEvents.
	void wait(WGPUFuture future, const bool *flag);
	// Lets pending callbacks run without blocking.
	void process_events();

	PyroWave::WebGPU::Pipeline create_pipeline(const char *label, const std::vector<const char *> &sources,
	                                           bool subgroups, bool dwt_shared, const char *entry_point,
	                                           const PyroWave::WebGPU::BindingLayout *bindings, size_t num_bindings,
	                                           const char *constant_name = nullptr, double constant_value = 0.0);
};
