// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT
#include "pyrowave_webgpu_common.hpp"
#include "pyrowave_webgpu_shaders.hpp"

#if defined(PYROWAVE_WEBGPU_WGPU_NATIVE)
#include <webgpu/wgpu.h>
#endif

#include <algorithm>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__wasi__) || defined(__EMSCRIPTEN__)
#include <sched.h>
static void yield_thread() { sched_yield(); }
#else
#include <thread>
static void yield_thread() { std::this_thread::yield(); }
#endif

namespace PyroWave
{
namespace WebGPU
{
WGPUStringView string_view(const char *str)
{
	WGPUStringView view = WGPU_STRING_VIEW_INIT;
	view.data = str;
	view.length = str ? strlen(str) : 0;
	return view;
}

std::string to_string(WGPUStringView view)
{
	if (!view.data)
		return {};
	if (view.length == WGPU_STRLEN)
		return view.data;
	return std::string(view.data, view.length);
}

void Pipeline::release()
{
	if (pipeline)
		wgpuComputePipelineRelease(pipeline);
	if (layout)
		wgpuBindGroupLayoutRelease(layout);
	pipeline = nullptr;
	layout = nullptr;
}

WGPUBuffer create_buffer(pyrowave_webgpu_device device, uint64_t size, WGPUBufferUsage usage, const char *label)
{
	WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
	desc.label = string_view(label);
	desc.usage = usage;
	// WebGPU wants copy sizes in multiples of 4.
	desc.size = (size + 3) & ~uint64_t(3);
	return wgpuDeviceCreateBuffer(device->device, &desc);
}

WGPUBindGroup create_bind_group(pyrowave_webgpu_device device, const Pipeline &pipeline,
                                const BindingResource *resources, size_t count)
{
	WGPUBindGroupEntry entries[8];
	if (count > 8)
		return nullptr;

	for (size_t i = 0; i < count; i++)
	{
		entries[i] = WGPU_BIND_GROUP_ENTRY_INIT;
		entries[i].binding = resources[i].binding;
		entries[i].buffer = resources[i].buffer;
		entries[i].offset = resources[i].offset;
		entries[i].size = resources[i].buffer ? resources[i].size : 0;
		entries[i].sampler = resources[i].sampler;
		entries[i].textureView = resources[i].view;
	}

	WGPUBindGroupDescriptor desc = {};
	desc.layout = pipeline.layout;
	desc.entryCount = count;
	desc.entries = entries;
	return wgpuDeviceCreateBindGroup(device->device, &desc);
}

void record_dispatches(WGPUComputePassEncoder pass, const std::vector<Dispatch> &dispatches)
{
	WGPUComputePipeline current = nullptr;
	for (auto &d : dispatches)
	{
		if (d.pipeline != current)
		{
			wgpuComputePassEncoderSetPipeline(pass, d.pipeline);
			current = d.pipeline;
		}
		wgpuComputePassEncoderSetBindGroup(pass, 0, d.bind_group, 0, nullptr);
		wgpuComputePassEncoderDispatchWorkgroups(pass, d.x, d.y, d.z);
	}
}

bool WaveletPyramid::init(pyrowave_webgpu_device device, const BlockLayout &layout)
{
	WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
	desc.label = string_view("wavelet-pyramid");
	desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc;
	desc.dimension = WGPUTextureDimension_2D;
	desc.size.width = uint32_t(layout.aligned_width / 2);
	desc.size.height = uint32_t(layout.aligned_height / 2);
	desc.size.depthOrArrayLayers = NumFrequencyBandsPerLevel * NumComponents;
	desc.format = WGPUTextureFormat_R32Float;
	desc.mipLevelCount = DecompositionLevels;
	desc.sampleCount = 1;
	texture = wgpuDeviceCreateTexture(device->device, &desc);
	if (!texture)
		return false;

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			WGPUTextureViewDescriptor view = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
			view.format = WGPUTextureFormat_R32Float;
			view.baseMipLevel = uint32_t(level);
			view.mipLevelCount = 1;
			view.baseArrayLayer = uint32_t(NumFrequencyBandsPerLevel * component);
			view.aspect = WGPUTextureAspect_All;

			view.dimension = WGPUTextureViewDimension_2DArray;
			view.arrayLayerCount = NumFrequencyBandsPerLevel;
			component_layer_views[component][level] = wgpuTextureCreateView(texture, &view);

			view.dimension = WGPUTextureViewDimension_2D;
			view.arrayLayerCount = 1;
			component_ll_views[component][level] = wgpuTextureCreateView(texture, &view);

			if (!component_layer_views[component][level] || !component_ll_views[component][level])
				return false;
		}
	}

	for (int level = 0; level < DecompositionLevels; level++)
	{
		WGPUTextureViewDescriptor view = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
		view.format = WGPUTextureFormat_R32Float;
		view.dimension = WGPUTextureViewDimension_2DArray;
		view.baseMipLevel = uint32_t(level);
		view.mipLevelCount = 1;
		view.baseArrayLayer = 0;
		view.arrayLayerCount = NumFrequencyBandsPerLevel * NumComponents;
		view.aspect = WGPUTextureAspect_All;
		level_views[level] = wgpuTextureCreateView(texture, &view);
		if (!level_views[level])
			return false;
	}

	WGPUTextureViewDescriptor view = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
	view.format = WGPUTextureFormat_R32Float;
	view.dimension = WGPUTextureViewDimension_2DArray;
	view.baseMipLevel = 0;
	view.mipLevelCount = DecompositionLevels;
	view.baseArrayLayer = 0;
	view.arrayLayerCount = NumFrequencyBandsPerLevel * NumComponents;
	view.aspect = WGPUTextureAspect_All;
	full_view = wgpuTextureCreateView(texture, &view);
	return full_view != nullptr;
}

void WaveletPyramid::dump(pyrowave_webgpu_device device, const BlockLayout &layout, const char *path)
{
	// Debug aid: every level and layer as raw floats, level by level.
	uint64_t total = 0;
	uint64_t offsets[DecompositionLevels];
	uint32_t row_pitch[DecompositionLevels];
	for (int level = 0; level < DecompositionLevels; level++)
	{
		offsets[level] = total;
		row_pitch[level] = (uint32_t(layout.level_width(level)) * 4 + 255) & ~255u;
		total += uint64_t(row_pitch[level]) * layout.level_height(level) * NumFrequencyBandsPerLevel * NumComponents;
	}
	WGPUBuffer buffer = create_buffer(device, total, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, "dump");
	WGPUCommandEncoder cmd = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
	for (int level = 0; level < DecompositionLevels; level++)
	{
		WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
		src.texture = texture;
		src.mipLevel = uint32_t(level);
		WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
		dst.buffer = buffer;
		dst.layout.offset = offsets[level];
		dst.layout.bytesPerRow = row_pitch[level];
		dst.layout.rowsPerImage = uint32_t(layout.level_height(level));
		WGPUExtent3D extent = { uint32_t(layout.level_width(level)), uint32_t(layout.level_height(level)),
		                        NumFrequencyBandsPerLevel * NumComponents };
		wgpuCommandEncoderCopyTextureToBuffer(cmd, &src, &dst, &extent);
	}
	WGPUCommandBuffer cb = wgpuCommandEncoderFinish(cmd, nullptr);
	wgpuQueueSubmit(device->queue, 1, &cb);
	wgpuCommandBufferRelease(cb);
	wgpuCommandEncoderRelease(cmd);

	Readback rb;
	rb.buffer = buffer;
	rb.size = (total + 3) & ~uint64_t(3);
	rb.map_async(device);
	if (rb.wait(device))
	{
		if (FILE *f = fopen(path, "wb"))
		{
			for (int level = 0; level < DecompositionLevels; level++)
			{
				uint32_t w = uint32_t(layout.level_width(level)), h = uint32_t(layout.level_height(level));
				for (uint32_t layer = 0; layer < NumFrequencyBandsPerLevel * NumComponents; layer++)
					for (uint32_t y = 0; y < h; y++)
						fwrite(static_cast<const uint8_t *>(rb.data) + offsets[level] +
						       (uint64_t(layer) * h + y) * row_pitch[level], 4, w, f);
			}
			fclose(f);
		}
	}
	rb.release();
}

void WaveletPyramid::release()
{
	for (auto &component_views : component_layer_views)
		for (auto &view : component_views)
			if (view)
				wgpuTextureViewRelease(view);
	for (auto &component_views : component_ll_views)
		for (auto &view : component_views)
			if (view)
				wgpuTextureViewRelease(view);
	for (auto &view : level_views)
		if (view)
			wgpuTextureViewRelease(view);
	if (full_view)
		wgpuTextureViewRelease(full_view);
	if (texture)
		wgpuTextureRelease(texture);
	*this = {};
}

bool BandTable::create(pyrowave_webgpu_device device, const char *label)
{
	buffer = create_buffer(device, ranges_offset() + ranges.size() * sizeof(uint32_t),
	                       WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, label);
	if (!buffer)
		return false;
	upload(device);
	return true;
}

void BandTable::upload(pyrowave_webgpu_device device)
{
	wgpuQueueWriteBuffer(device->queue, buffer, 0, registers.data(), registers.size());
	wgpuQueueWriteBuffer(device->queue, buffer, ranges_offset(), ranges.data(), ranges.size() * sizeof(uint32_t));
}

Dispatch BandTable::dispatch(WGPUComputePipeline pipeline, WGPUBindGroup group) const
{
	// The shader numbers workgroups x + y * num_workgroups.x, so large tables can
	// spill into y.
	Dispatch d = {};
	d.pipeline = pipeline;
	d.bind_group = group;
	d.x = std::min<uint32_t>(total_workgroups, 32768u);
	d.y = (total_workgroups + d.x - 1) / d.x;
	d.z = 1;
	return d;
}

void BandTable::release()
{
	if (buffer)
		wgpuBufferRelease(buffer);
	*this = {};
}

bool StageTimer::init(pyrowave_webgpu_device device, int stages)
{
	num_stages = stages;
	if (!device->timestamps || stages > 8)
		return true;

	WGPUQuerySetDescriptor desc = {};
	desc.label = string_view("stage-timestamps");
	desc.type = WGPUQueryType_Timestamp;
	desc.count = uint32_t(stages * 2);
	query_set = wgpuDeviceCreateQuerySet(device->device, &desc);
	if (!query_set)
		return false;

	resolve_buffer = create_buffer(device, readback_size(),
	                               WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc, "timestamp-resolve");
	if (!resolve_buffer)
		return false;

	for (int i = 0; i < stages; i++)
	{
		stage_writes[i].querySet = query_set;
		stage_writes[i].beginningOfPassWriteIndex = uint32_t(2 * i);
		stage_writes[i].endOfPassWriteIndex = uint32_t(2 * i + 1);
	}

	return true;
}

const WGPUPassTimestampWrites *StageTimer::writes(int stage)
{
	return query_set ? &stage_writes[stage] : nullptr;
}

void StageTimer::resolve(WGPUCommandEncoder cmd, WGPUBuffer readback, uint64_t readback_offset)
{
	if (!query_set)
		return;
	wgpuCommandEncoderResolveQuerySet(cmd, query_set, 0, uint32_t(num_stages * 2), resolve_buffer, 0);
	wgpuCommandEncoderCopyBufferToBuffer(cmd, resolve_buffer, 0, readback, readback_offset, readback_size());
}

void StageTimer::read(const void *mapped, pyrowave_webgpu_timings *timings) const
{
	*timings = {};
	timings->num_stages = num_stages;
	if (!query_set || !mapped)
		return;

	uint64_t ts[16];
	memcpy(ts, mapped, readback_size());
	for (int i = 0; i < num_stages; i++)
	{
		uint64_t ns = ts[2 * i + 1] > ts[2 * i] ? ts[2 * i + 1] - ts[2 * i] : 0;
		timings->stage_ns[i] = ns;
	}
	if (ts[2 * num_stages - 1] > ts[0])
		timings->total_ns = ts[2 * num_stages - 1] - ts[0];
}

void StageTimer::release()
{
	if (query_set)
		wgpuQuerySetRelease(query_set);
	if (resolve_buffer)
		wgpuBufferRelease(resolve_buffer);
	*this = {};
}

bool Readback::ensure_size(pyrowave_webgpu_device device, uint64_t size_, const char *label)
{
	size_ = (size_ + 3) & ~uint64_t(3);
	if (buffer && size >= size_)
		return true;

	unmap();
	if (buffer)
		wgpuBufferRelease(buffer);

	size = size_;
	buffer = create_buffer(device, size, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, label);
	return buffer != nullptr;
}

static void on_buffer_mapped(WGPUMapAsyncStatus status, WGPUStringView message, void *userdata1, void *userdata2)
{
	auto *readback = static_cast<Readback *>(userdata1);
	auto *device = static_cast<pyrowave_webgpu_device>(userdata2);
	readback->pending = false;
	readback->completed = true;

	if (status == WGPUMapAsyncStatus_Success)
	{
		readback->mapped = true;
		readback->data = wgpuBufferGetConstMappedRange(readback->buffer, 0, size_t(readback->size));
		readback->failed = readback->data == nullptr;
	}
	else
	{
		readback->failed = true;
		device->log("Buffer readback failed: %s\n", to_string(message).c_str());
	}
}

void Readback::map_async(pyrowave_webgpu_device device)
{
	pending = true;
	completed = false;
	mapped = false;
	failed = false;
	data = nullptr;

	WGPUBufferMapCallbackInfo info = {};
	info.mode = WGPUCallbackMode_AllowProcessEvents;
	info.callback = on_buffer_mapped;
	info.userdata1 = this;
	info.userdata2 = device;
	future = wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, size_t(size), info);
}

bool Readback::wait(pyrowave_webgpu_device device)
{
	if (pending)
		device->wait(future, &completed);
	return mapped && !failed;
}

void Readback::unmap()
{
	if (mapped)
		wgpuBufferUnmap(buffer);
	mapped = false;
	data = nullptr;
}

void Readback::release()
{
	unmap();
	if (buffer)
		wgpuBufferRelease(buffer);
	*this = {};
}

bool create_encode_pipelines(pyrowave_webgpu_device device)
{
	if (device->encode_pipelines_ready)
		return true;

	if (device->subgroup_min_size && (device->subgroup_min_size < 16 || device->subgroup_max_size > 64))
	{
		// The encoder's analysis and packing work on 16 lane clusters inside a 64 lane
		// workgroup, the same range the Vulkan encoder asks subgroup size control for.
		device->log("Encoder needs a subgroup size between 16 and 64, device reports %u to %u.\n",
		            device->subgroup_min_size, device->subgroup_max_size);
		return false;
	}

	using B = BindingType;

	for (int dc_shift = 0; dc_shift < 2; dc_shift++)
	{
		static const BindingLayout layout[] = {
			{ 0, B::Uniform }, { 1, B::Texture2D }, { 2, B::Sampler }, { 3, B::StorageTexture2DArray },
		};
		device->dwt[dc_shift] = device->create_pipeline(
				dc_shift ? "dwt-dc-shift" : "dwt", { wgsl_common, wgsl_dwt_common, wgsl_dwt },
				false, true, "main", layout, 4, "DCShift", double(dc_shift));
		if (!device->dwt[dc_shift].pipeline)
			return false;
	}

	{
		static const BindingLayout layout[] = {
			{ 0, B::StorageRead }, { 1, B::Texture2DArray },
			{ 2, B::StorageReadWrite }, { 3, B::StorageReadWrite }, { 4, B::StorageReadWrite }, { 7, B::StorageRead },
		};
		device->quant = device->create_pipeline(
				"wavelet-quant", { wgsl_common, wgsl_subgroup, wgsl_band_dispatch, wgsl_wavelet_quant },
				true, false, "main", layout, 6);
		if (!device->quant.pipeline)
			return false;
	}

	{
		static const BindingLayout layout[] = {
			{ 0, B::StorageRead }, { 1, B::StorageReadWrite }, { 2, B::StorageRead }, { 7, B::StorageRead },
		};
		device->analyze = device->create_pipeline(
				"analyze-rate-control", { wgsl_subgroup, wgsl_band_dispatch, wgsl_analyze_rate_control },
				true, false, "main", layout, 4);
		if (!device->analyze.pipeline)
			return false;
	}

	{
		static const BindingLayout layout[] = { { 0, B::StorageReadWrite } };
		device->analyze_finalize = device->create_pipeline(
				"analyze-rate-control-finalize", { wgsl_analyze_rate_control_finalize },
				false, false, "main", layout, 1);
		if (!device->analyze_finalize.pipeline)
			return false;
	}

	{
		static const BindingLayout layout[] = {
			{ 0, B::Uniform }, { 1, B::StorageRead }, { 2, B::StorageReadWrite },
		};
		device->resolve = device->create_pipeline(
				"resolve-rate-control", { wgsl_subgroup, wgsl_resolve_rate_control },
				true, false, "main", layout, 3);
		if (!device->resolve.pipeline)
			return false;
	}

	{
		static const BindingLayout layout[] = {
			{ 0, B::StorageRead }, { 1, B::StorageReadWrite }, { 2, B::StorageReadWrite }, { 3, B::StorageRead },
			{ 4, B::StorageReadWrite }, { 5, B::StorageRead }, { 6, B::StorageRead }, { 7, B::StorageRead },
		};
		device->block_packing = device->create_pipeline(
				"block-packing", { wgsl_common, wgsl_subgroup, wgsl_band_dispatch, wgsl_block_packing },
				true, false, "main", layout, 8);
		if (!device->block_packing.pipeline)
			return false;
	}

	device->encode_pipelines_ready = true;
	return true;
}

static bool create_decode_pipelines(pyrowave_webgpu_device device)
{
	using B = BindingType;

	{
		static const BindingLayout layout[] = {
			{ 0, B::StorageRead }, { 1, B::StorageTexture2DArray }, { 2, B::StorageRead }, { 3, B::StorageRead },
			{ 7, B::StorageRead },
		};
		device->dequant = device->create_pipeline(
				"wavelet-dequant", { wgsl_common, wgsl_subgroup, wgsl_band_dispatch, wgsl_wavelet_dequant },
				true, false, "main", layout, 5);
		if (!device->dequant.pipeline)
			return false;
	}

	{
		static const BindingLayout layout[] = {
			{ 0, B::Uniform }, { 1, B::Texture2DArray }, { 2, B::Sampler }, { 3, B::StorageTexture2D },
		};
		device->idwt = device->create_pipeline(
				"idwt", { wgsl_common, wgsl_dwt_common, wgsl_idwt },
				false, true, "main", layout, 4);
		if (!device->idwt.pipeline)
			return false;
	}

	{
		static const BindingLayout layout[] = {
			{ 0, B::Uniform }, { 1, B::Texture2DArray }, { 2, B::Sampler }, { 4, B::StorageReadWrite },
		};
		device->idwt_final = device->create_pipeline(
				"idwt-final", { wgsl_common, wgsl_dwt_common, wgsl_idwt },
				false, true, "main_final", layout, 4);
		if (!device->idwt_final.pipeline)
			return false;
	}

	return true;
}
}
}

using namespace PyroWave;
using namespace PyroWave::WebGPU;

void pyrowave_webgpu_device_opaque::log(const char *fmt, ...) const
{
	char buffer[2048];
	va_list va;
	va_start(va, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);

	if (message_cb)
		message_cb(message_userdata, buffer);
	else
		fprintf(stderr, "pyrowave-webgpu: %s", buffer);
}

void pyrowave_webgpu_device_opaque::process_events()
{
	wgpuInstanceProcessEvents(instance);
}

void pyrowave_webgpu_device_opaque::wait(WGPUFuture future, const bool *flag)
{
	if (timed_wait_any)
	{
		WGPUFutureWaitInfo info = {};
		info.future = future;
		while (!*flag)
		{
			WGPUWaitStatus status = wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX);
			if (status != WGPUWaitStatus_Success && status != WGPUWaitStatus_TimedOut)
				break;
		}
	}

	// Spinning on ProcessEvents is the only portable way to wait without TimedWaitAny.
	while (!*flag && !device_lost)
	{
		wgpuInstanceProcessEvents(instance);
		if (!*flag)
			yield_thread();
	}
}

namespace
{
struct ErrorScopeResult
{
	bool done = false;
	WGPUErrorType type = WGPUErrorType_NoError;
	std::string message;
};

void on_pop_error_scope(WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void *userdata1, void *)
{
	auto *result = static_cast<ErrorScopeResult *>(userdata1);
	result->type = type;
	result->message = to_string(message);
	result->done = true;
}
}

Pipeline pyrowave_webgpu_device_opaque::create_pipeline(
		const char *label, const std::vector<const char *> &sources,
		bool subgroups, bool dwt_shared, const char *entry_point,
		const BindingLayout *bindings, size_t num_bindings,
		const char *constant_name, double constant_value)
{
	std::string source;

	if (subgroups)
	{
		// The shaders run subgroup operations under control flow that is only uniform
		// per cluster of lanes, as the GLSL does. See subgroup.wgsl.
		if (wgsl_enable_subgroups)
			source += "enable subgroups;\n";
		source += "diagnostic(off, subgroup_uniformity);\n";
	}

	if (dwt_shared)
	{
		// dwt_common.wgsl: the shared memory tile type depends on the precision.
		if (precision == 2)
		{
			source += "alias SHARED_VEC2 = vec2<f32>;\n"
			          "fn shared_pack(v: vec2<f32>) -> SHARED_VEC2 { return v; }\n"
			          "fn shared_unpack(v: SHARED_VEC2) -> vec2<f32> { return v; }\n";
		}
		else
		{
			// pack2x16float does not promise how it rounds, and on D3D12 it does not round
			// to nearest, which costs about 0.1 dB. Round first; exact values pack exactly.
			source += "alias SHARED_VEC2 = u32;\n"
			          "fn shared_pack(v: vec2<f32>) -> SHARED_VEC2 {\n"
			          "    return pack2x16float(round_to_f16_vec4(vec4<f32>(v, 0.0, 0.0)).xy);\n"
			          "}\n"
			          "fn shared_unpack(v: SHARED_VEC2) -> vec2<f32> { return unpack2x16float(v); }\n";
		}
	}

	for (auto *s : sources)
		source += s;

	Pipeline result;
	ErrorScopeResult scope;
	wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);

	WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
	wgsl.code = string_view(source.c_str());
	WGPUShaderModuleDescriptor module_desc = {};
	module_desc.nextInChain = &wgsl.chain;
	module_desc.label = string_view(label);
	WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &module_desc);

	WGPUBindGroupLayoutEntry entries[8];
	for (size_t i = 0; i < num_bindings; i++)
	{
		auto &entry = entries[i];
		entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
		entry.binding = bindings[i].binding;
		entry.visibility = WGPUShaderStage_Compute;

		switch (bindings[i].type)
		{
		case BindingType::Uniform:
			entry.buffer.type = WGPUBufferBindingType_Uniform;
			break;
		case BindingType::StorageRead:
			entry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
			break;
		case BindingType::StorageReadWrite:
			entry.buffer.type = WGPUBufferBindingType_Storage;
			break;
		case BindingType::Texture2D:
		case BindingType::Texture2DArray:
			entry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
			entry.texture.viewDimension = bindings[i].type == BindingType::Texture2D ?
			                              WGPUTextureViewDimension_2D : WGPUTextureViewDimension_2DArray;
			break;
		case BindingType::Sampler:
			entry.sampler.type = WGPUSamplerBindingType_NonFiltering;
			break;
		case BindingType::StorageTexture2D:
		case BindingType::StorageTexture2DArray:
			entry.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
			entry.storageTexture.format = WGPUTextureFormat_R32Float;
			entry.storageTexture.viewDimension = bindings[i].type == BindingType::StorageTexture2D ?
			                                     WGPUTextureViewDimension_2D : WGPUTextureViewDimension_2DArray;
			break;
		}
	}

	WGPUBindGroupLayoutDescriptor bgl_desc = {};
	bgl_desc.label = string_view(label);
	bgl_desc.entryCount = num_bindings;
	bgl_desc.entries = entries;
	result.layout = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

	WGPUPipelineLayoutDescriptor layout_desc = {};
	layout_desc.label = string_view(label);
	layout_desc.bindGroupLayoutCount = 1;
	layout_desc.bindGroupLayouts = &result.layout;
	WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);

	WGPUConstantEntry constant = {};
	WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
	pipeline_desc.label = string_view(label);
	pipeline_desc.layout = pipeline_layout;
	pipeline_desc.compute.module = module;
	pipeline_desc.compute.entryPoint = string_view(entry_point);
	if (constant_name)
	{
		constant.key = string_view(constant_name);
		constant.value = constant_value;
		pipeline_desc.compute.constantCount = 1;
		pipeline_desc.compute.constants = &constant;
	}
	result.pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

	WGPUPopErrorScopeCallbackInfo pop_info = {};
	pop_info.mode = WGPUCallbackMode_AllowProcessEvents;
	pop_info.callback = on_pop_error_scope;
	pop_info.userdata1 = &scope;
	WGPUFuture future = wgpuDevicePopErrorScope(device, pop_info);
	wait(future, &scope.done);

	if (pipeline_layout)
		wgpuPipelineLayoutRelease(pipeline_layout);
	if (module)
		wgpuShaderModuleRelease(module);

	if (scope.type != WGPUErrorType_NoError || !result.pipeline || !result.layout)
	{
		result.release();

		if (subgroups && wgsl_enable_subgroups)
		{
			// Some WGSL front ends reject the standard `enable subgroups;` but accept the
			// builtins without it (naga as of wgpu 29, and so any host built on it).
			// Try once without, and stick with that if it works.
			wgsl_enable_subgroups = false;
			result = create_pipeline(label, sources, subgroups, dwt_shared, entry_point,
			                         bindings, num_bindings, constant_name, constant_value);
			if (result.pipeline)
				return result;
			wgsl_enable_subgroups = true;
		}

		log("Failed to create pipeline %s: %s\n", label, scope.message.c_str());
	}

	return result;
}

namespace
{
struct AdapterRequest
{
	bool done = false;
	WGPUAdapter adapter = nullptr;
	std::string message;
};

struct DeviceRequest
{
	bool done = false;
	WGPUDevice device = nullptr;
	std::string message;
};

void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void *userdata1, void *)
{
	auto *request = static_cast<AdapterRequest *>(userdata1);
	if (status == WGPURequestAdapterStatus_Success)
		request->adapter = adapter;
	request->message = to_string(message);
	request->done = true;
}

void on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void *userdata1, void *)
{
	auto *request = static_cast<DeviceRequest *>(userdata1);
	if (status == WGPURequestDeviceStatus_Success)
		request->device = device;
	request->message = to_string(message);
	request->done = true;
}

void on_uncaptured_error(const WGPUDevice *, WGPUErrorType type, WGPUStringView message, void *userdata1, void *)
{
	auto *device = static_cast<pyrowave_webgpu_device>(userdata1);
	device->log("WebGPU error %d: %s\n", int(type), to_string(message).c_str());
}

void on_device_lost(const WGPUDevice *, WGPUDeviceLostReason reason, WGPUStringView message, void *userdata1, void *)
{
	auto *device = static_cast<pyrowave_webgpu_device>(userdata1);
	if (!device)
		return;
	device->device_lost = true;
	if (reason != WGPUDeviceLostReason_Destroyed && reason != WGPUDeviceLostReason_CallbackCancelled)
		device->log("WebGPU device lost: %s\n", to_string(message).c_str());
}

WGPUBackendType backend_from_env(WGPUBackendType fallback)
{
	const char *env = getenv("PYROWAVE_WEBGPU_BACKEND");
	if (!env)
		return fallback;
	if (strcmp(env, "vulkan") == 0)
		return WGPUBackendType_Vulkan;
	if (strcmp(env, "d3d12") == 0)
		return WGPUBackendType_D3D12;
	if (strcmp(env, "metal") == 0)
		return WGPUBackendType_Metal;
	return fallback;
}

const char *backend_name(WGPUBackendType type)
{
	switch (type)
	{
	case WGPUBackendType_Vulkan: return "Vulkan";
	case WGPUBackendType_D3D12: return "D3D12";
	case WGPUBackendType_D3D11: return "D3D11";
	case WGPUBackendType_Metal: return "Metal";
	case WGPUBackendType_OpenGL: return "OpenGL";
	case WGPUBackendType_OpenGLES: return "OpenGLES";
	case WGPUBackendType_WebGPU: return "WebGPU";
	default: return "unknown";
	}
}

void destroy_device(pyrowave_webgpu_device device)
{
	device->dequant.release();
	device->idwt.release();
	device->idwt_final.release();
	for (auto &p : device->dwt)
		p.release();
	device->quant.release();
	device->analyze.release();
	device->analyze_finalize.release();
	device->resolve.release();
	device->block_packing.release();
	if (device->mirror_repeat_sampler)
		wgpuSamplerRelease(device->mirror_repeat_sampler);
	if (device->queue)
		wgpuQueueRelease(device->queue);
	if (device->device)
		wgpuDeviceRelease(device->device);
	if (device->adapter)
		wgpuAdapterRelease(device->adapter);
	if (device->instance)
		wgpuInstanceRelease(device->instance);
	delete device;
}
}

void pyrowave_webgpu_get_api_version(uint32_t *major, uint32_t *minor, uint32_t *patch)
{
	*major = PYROWAVE_WEBGPU_API_VERSION_MAJOR;
	*minor = PYROWAVE_WEBGPU_API_VERSION_MINOR;
	*patch = PYROWAVE_WEBGPU_API_VERSION_PATCH;
}

const char *pyrowave_webgpu_result_to_string(pyrowave_webgpu_result result)
{
	switch (result)
	{
	case PYROWAVE_WEBGPU_SUCCESS: return "success";
	case PYROWAVE_WEBGPU_ERROR_GENERIC: return "generic error";
	case PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT: return "invalid argument";
	case PYROWAVE_WEBGPU_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
	case PYROWAVE_WEBGPU_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
	case PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE: return "unsupported device";
	case PYROWAVE_WEBGPU_ERROR_SHADER_COMPILATION: return "shader compilation failed";
	case PYROWAVE_WEBGPU_ERROR_CORRUPT_BITSTREAM: return "corrupt bitstream";
	case PYROWAVE_WEBGPU_NOT_READY: return "not ready";
	default: return "unknown result";
	}
}

pyrowave_webgpu_result pyrowave_webgpu_device_create(const pyrowave_webgpu_device_create_info *info,
                                                     pyrowave_webgpu_device *out_device)
{
	if (!info || !out_device)
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	bool external = info->device != nullptr;
	if (external && (!info->instance || !info->adapter))
		return PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT;

	auto *device = new pyrowave_webgpu_device_opaque;
	device->message_cb = info->message_callback;
	device->message_userdata = info->message_userdata;

	if (const char *env = getenv("PYROWAVE_PRECISION"))
	{
		int precision = int(strtol(env, nullptr, 0));
		if (precision == 2 || precision == 1)
			device->precision = precision;
		else if (precision == 0)
			device->log("PYROWAVE_PRECISION=0 (FP16 math) is not implemented, using 1.\n");
		else
			device->log("PYROWAVE_PRECISION must be in range [0, 2].\n");
	}

	if (external)
	{
		device->instance = info->instance;
		device->adapter = info->adapter;
		device->device = info->device;
		wgpuInstanceAddRef(device->instance);
		wgpuAdapterAddRef(device->adapter);
		wgpuDeviceAddRef(device->device);
		device->timed_wait_any = info->timed_wait_any;

		bool has_subgroups = wgpuDeviceHasFeature(device->device, WGPUFeatureName_Subgroups) != 0;
#if defined(PYROWAVE_WEBGPU_WGPU_NATIVE)
		if (!has_subgroups && wgpuDeviceHasFeature(device->device, WGPUFeatureName(WGPUNativeFeature_Subgroup)))
		{
			has_subgroups = true;
			device->wgsl_enable_subgroups = false;
		}
#endif
		if (!has_subgroups)
		{
			device->log("The device was not created with the subgroups feature.\n");
			destroy_device(device);
			return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;
		}
		device->timestamps = wgpuDeviceHasFeature(device->device, WGPUFeatureName_TimestampQuery) != 0;
	}
	else
	{
		WGPUInstanceDescriptor instance_desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
		WGPUInstanceFeatureName timed_wait = WGPUInstanceFeatureName_TimedWaitAny;
		// Opt in rather than probed: wgpu-native v29 panics in wgpuHasInstanceFeature.
		const char *wait_any_env = getenv("PYROWAVE_WEBGPU_TIMED_WAIT_ANY");
		if (info->timed_wait_any || (wait_any_env && strcmp(wait_any_env, "1") == 0))
		{
			instance_desc.requiredFeatureCount = 1;
			instance_desc.requiredFeatures = &timed_wait;
			device->timed_wait_any = true;
		}
		device->instance = wgpuCreateInstance(&instance_desc);
		if (!device->instance)
		{
			destroy_device(device);
			return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;
		}

		WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
		options.featureLevel = WGPUFeatureLevel_Core;
		options.powerPreference = WGPUPowerPreference_HighPerformance;
		options.backendType = backend_from_env(info->backend_type);

		AdapterRequest adapter_request;
		WGPURequestAdapterCallbackInfo adapter_cb = {};
		adapter_cb.mode = WGPUCallbackMode_AllowProcessEvents;
		adapter_cb.callback = on_adapter;
		adapter_cb.userdata1 = &adapter_request;
		device->wait(wgpuInstanceRequestAdapter(device->instance, &options, adapter_cb), &adapter_request.done);
		device->adapter = adapter_request.adapter;

		if (!device->adapter)
		{
			device->log("No WebGPU adapter: %s\n", adapter_request.message.c_str());
			destroy_device(device);
			return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;
		}

		WGPUFeatureName subgroups = WGPUFeatureName_Subgroups;
#if defined(PYROWAVE_WEBGPU_WGPU_NATIVE)
		// wgpu-native (as of v29) does not report the standard feature, only its own
		// native one, which enables the same WGSL. This and the matching check for
		// application provided devices are the only places the backend looks past the
		// standard header.
		if (!wgpuAdapterHasFeature(device->adapter, subgroups) &&
		    wgpuAdapterHasFeature(device->adapter, WGPUFeatureName(WGPUNativeFeature_Subgroup)))
		{
			subgroups = WGPUFeatureName(WGPUNativeFeature_Subgroup);
			device->wgsl_enable_subgroups = false;
		}
#endif

		if (!wgpuAdapterHasFeature(device->adapter, subgroups))
		{
			device->log("The adapter does not support the subgroups feature.\n");
			destroy_device(device);
			return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;
		}

		WGPUFeatureName features[2] = { subgroups };
		size_t num_features = 1;
		if (info->enable_timestamps && wgpuAdapterHasFeature(device->adapter, WGPUFeatureName_TimestampQuery))
		{
			features[num_features++] = WGPUFeatureName_TimestampQuery;
			device->timestamps = true;
		}

		// Large frames need more than the default 128 MiB storage binding for the
		// quantizer's payload scratch buffer. Ask for whatever the adapter has.
		WGPULimits adapter_limits = WGPU_LIMITS_INIT;
		wgpuAdapterGetLimits(device->adapter, &adapter_limits);
		WGPULimits limits = WGPU_LIMITS_INIT;
		limits.maxStorageBufferBindingSize = adapter_limits.maxStorageBufferBindingSize;
		limits.maxBufferSize = adapter_limits.maxBufferSize;

		WGPUDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_INIT;
		device_desc.label = string_view("pyrowave");
		device_desc.requiredFeatureCount = num_features;
		device_desc.requiredFeatures = features;
		device_desc.requiredLimits = &limits;
		device_desc.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;
		device_desc.uncapturedErrorCallbackInfo.userdata1 = device;
		device_desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowProcessEvents;
		device_desc.deviceLostCallbackInfo.callback = on_device_lost;
		device_desc.deviceLostCallbackInfo.userdata1 = device;

		DeviceRequest device_request;
		WGPURequestDeviceCallbackInfo device_cb = {};
		device_cb.mode = WGPUCallbackMode_AllowProcessEvents;
		device_cb.callback = on_device;
		device_cb.userdata1 = &device_request;
		device->wait(wgpuAdapterRequestDevice(device->adapter, &device_desc, device_cb), &device_request.done);
		device->device = device_request.device;

		if (!device->device)
		{
			device->log("Failed to create WebGPU device: %s\n", device_request.message.c_str());
			destroy_device(device);
			return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;
		}
	}

	WGPUAdapterInfo adapter_info = WGPU_ADAPTER_INFO_INIT;
	if (wgpuAdapterGetInfo(device->adapter, &adapter_info) == WGPUStatus_Success)
	{
		device->subgroup_min_size = adapter_info.subgroupMinSize;
		device->subgroup_max_size = adapter_info.subgroupMaxSize;
		device->log("Using %s (%s), subgroup size %u to %u, precision %d%s.\n",
		            to_string(adapter_info.device).c_str(), backend_name(adapter_info.backendType),
		            adapter_info.subgroupMinSize, adapter_info.subgroupMaxSize, device->precision,
		            device->timestamps ? ", timestamps" : "");
		wgpuAdapterInfoFreeMembers(adapter_info);
	}

	if (device->subgroup_max_size > 128)
	{
		device->log("Subgroups wider than 128 lanes are not supported.\n");
		destroy_device(device);
		return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;
	}

	WGPULimits device_limits = WGPU_LIMITS_INIT;
	if (wgpuDeviceGetLimits(device->device, &device_limits) == WGPUStatus_Success &&
	    device_limits.minUniformBufferOffsetAlignment > UniformSlotSize)
	{
		destroy_device(device);
		return PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE;
	}

	device->queue = wgpuDeviceGetQueue(device->device);

	WGPUSamplerDescriptor sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
	sampler_desc.label = string_view("mirror-repeat");
	sampler_desc.addressModeU = WGPUAddressMode_MirrorRepeat;
	sampler_desc.addressModeV = WGPUAddressMode_MirrorRepeat;
	sampler_desc.addressModeW = WGPUAddressMode_MirrorRepeat;
	sampler_desc.magFilter = WGPUFilterMode_Nearest;
	sampler_desc.minFilter = WGPUFilterMode_Nearest;
	sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
	device->mirror_repeat_sampler = wgpuDeviceCreateSampler(device->device, &sampler_desc);

	if (!create_decode_pipelines(device))
	{
		destroy_device(device);
		return PYROWAVE_WEBGPU_ERROR_SHADER_COMPILATION;
	}

	*out_device = device;
	return PYROWAVE_WEBGPU_SUCCESS;
}

void pyrowave_webgpu_device_destroy(pyrowave_webgpu_device device)
{
	if (device)
		destroy_device(device);
}

void pyrowave_webgpu_device_get_handles(pyrowave_webgpu_device device,
                                        WGPUInstance *instance, WGPUAdapter *adapter, WGPUDevice *device_handle)
{
	if (instance)
		*instance = device->instance;
	if (adapter)
		*adapter = device->adapter;
	if (device_handle)
		*device_handle = device->device;
}
