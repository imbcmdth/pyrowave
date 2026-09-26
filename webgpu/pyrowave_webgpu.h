// Copyright (c) 2026 Hans-Kristian Arntzen and PyroWave contributors
// SPDX-License-Identifier: MIT

#ifndef PYROWAVE_WEBGPU_H_
#define PYROWAVE_WEBGPU_H_

// WebGPU encoder and decoder for PyroWave, written against the standard webgpu.h
// (https://github.com/webgpu-native/webgpu-headers). It needs only the standard
// "subgroups" feature, and uses "timestamp-query" when asked to and available.
//
// The bitstream is the one described in bitstream/bitstream.md, so this interoperates
// with the Vulkan and Metal implementations in either direction.
//
// Everything is prefixed pyrowave_webgpu_ so this header can be used next to
// pyrowave.h in the same program.
//
// Threading: none of the entry points are thread safe. Use an encoder or decoder
// from one thread at a time, and do not share a device between threads without
// external locking.

#include <stddef.h>
#include <stdint.h>
#include <webgpu/webgpu.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

#define PYROWAVE_WEBGPU_API_VERSION_MAJOR 0
#define PYROWAVE_WEBGPU_API_VERSION_MINOR 1
#define PYROWAVE_WEBGPU_API_VERSION_PATCH 0

#if !defined(PYROWAVE_WEBGPU_PUBLIC_API)
#if defined(PYROWAVE_WEBGPU_EXPORT_SYMBOLS)
#if defined(__GNUC__)
#define PYROWAVE_WEBGPU_PUBLIC_API __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define PYROWAVE_WEBGPU_PUBLIC_API __declspec(dllexport)
#else
#define PYROWAVE_WEBGPU_PUBLIC_API
#endif
#else
#define PYROWAVE_WEBGPU_PUBLIC_API
#endif
#endif

// Codes 0 through -7 carry the same meaning as in pyrowave_metal.h.
typedef enum pyrowave_webgpu_result
{
	PYROWAVE_WEBGPU_SUCCESS = 0,
	PYROWAVE_WEBGPU_ERROR_GENERIC = -1,
	PYROWAVE_WEBGPU_ERROR_INVALID_ARGUMENT = -2,
	PYROWAVE_WEBGPU_ERROR_OUT_OF_HOST_MEMORY = -3,
	PYROWAVE_WEBGPU_ERROR_OUT_OF_DEVICE_MEMORY = -4,
	PYROWAVE_WEBGPU_ERROR_UNSUPPORTED_DEVICE = -5,
	PYROWAVE_WEBGPU_ERROR_SHADER_COMPILATION = -6,
	PYROWAVE_WEBGPU_ERROR_CORRUPT_BITSTREAM = -7,
	// Only from the non-blocking poll functions: the GPU has not finished yet.
	PYROWAVE_WEBGPU_NOT_READY = 1,
	PYROWAVE_WEBGPU_RESULT_INT_MAX = 0x7fffffff
} pyrowave_webgpu_result;

typedef enum pyrowave_webgpu_chroma_subsampling
{
	PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_420 = 0,
	PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_444 = 1,
	PYROWAVE_WEBGPU_CHROMA_SUBSAMPLING_INT_MAX = 0x7fffffff
} pyrowave_webgpu_chroma_subsampling;

typedef struct pyrowave_webgpu_device_opaque *pyrowave_webgpu_device;
typedef struct pyrowave_webgpu_encoder_opaque *pyrowave_webgpu_encoder;
typedef struct pyrowave_webgpu_decoder_opaque *pyrowave_webgpu_decoder;

typedef void (*pyrowave_webgpu_message_cb)(void *userdata, const char *msg);

PYROWAVE_WEBGPU_PUBLIC_API void
pyrowave_webgpu_get_api_version(uint32_t *major, uint32_t *minor, uint32_t *patch);

// Never returns NULL.
PYROWAVE_WEBGPU_PUBLIC_API const char *
pyrowave_webgpu_result_to_string(pyrowave_webgpu_result result);

// Device API.

typedef struct pyrowave_webgpu_device_create_info
{
	// Either leave all three NULL and PyroWave creates its own instance, adapter and
	// device, or pass in all three to share the application's. A device passed in
	// must have been created with WGPUFeatureName_Subgroups enabled. PyroWave takes
	// its own references and releases them on destroy.
	//
	// The instance is needed to wait for GPU work: PyroWave uses
	// wgpuInstanceWaitAny if the instance was created with
	// WGPUInstanceFeatureName_TimedWaitAny, and otherwise spins on
	// wgpuInstanceProcessEvents.
	WGPUInstance instance;
	WGPUAdapter adapter;
	WGPUDevice device;

	// Only used when PyroWave creates the adapter. WGPUBackendType_Undefined lets the
	// implementation choose. Can be overridden with PYROWAVE_WEBGPU_BACKEND
	// (vulkan, d3d12, metal).
	WGPUBackendType backend_type;

	// Only used when PyroWave creates the device: also request timestamp queries,
	// so that the encoder and decoder can report GPU time per stage.
	bool enable_timestamps;

	// Block in wgpuInstanceWaitAny instead of spinning on wgpuInstanceProcessEvents
	// when waiting for the GPU. If PyroWave creates the instance, it asks for
	// WGPUInstanceFeatureName_TimedWaitAny; an instance passed in must have been
	// created with it. Off by default because not every implementation has it
	// (wgpu-native v29 does not). PYROWAVE_WEBGPU_TIMED_WAIT_ANY=1 also turns it on.
	bool timed_wait_any;

	// Optional message callback, or NULL to print to stderr.
	pyrowave_webgpu_message_cb message_callback;
	void *message_userdata;
} pyrowave_webgpu_device_create_info;

// Creates the shared device object and compiles the decode pipelines. The encode
// pipelines are compiled by the first encoder created.
//
// The PYROWAVE_PRECISION environment variable picks the wavelet precision, as on
// the Vulkan side: 1 (the default) stores the two highest resolution levels and the
// transform's shared memory tile as FP16, and 2 is FP32 throughout. 0 (FP16 math) is
// not implemented and falls back to 1, as the Vulkan build does on devices without
// FP16 arithmetic.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_device_create(const pyrowave_webgpu_device_create_info *info, pyrowave_webgpu_device *device);

// All encoders and decoders created from this device must be destroyed first.
PYROWAVE_WEBGPU_PUBLIC_API void
pyrowave_webgpu_device_destroy(pyrowave_webgpu_device device);

// The handles PyroWave is using, for applications that want to put their own GPU
// work next to it. Not reference counted; valid while the device lives.
PYROWAVE_WEBGPU_PUBLIC_API void
pyrowave_webgpu_device_get_handles(pyrowave_webgpu_device device,
                                   WGPUInstance *instance, WGPUAdapter *adapter, WGPUDevice *device_handle);

// Shared types.

typedef enum pyrowave_webgpu_cpu_buffer_format
{
	PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_NV12 = 0,    // 2 planes. Y in 8bpp, then CbCr interleaved in 16bpp. Encode only.
	PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV420P = 1, // 3 planes, half resolution chroma.
	PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_YUV444P = 2, // 3 planes, full resolution chroma.
	PYROWAVE_WEBGPU_CPU_BUFFER_FORMAT_INT_MAX = 0x7fffffff
} pyrowave_webgpu_cpu_buffer_format;

typedef struct pyrowave_webgpu_cpu_buffer
{
	// Written in decoder, read-only in encoder.
	void *data[3];
	// Must be at least width for plane times texel size of the plane.
	size_t row_stride_in_bytes[3];
	// Must be at least row_stride times height of plane.
	size_t plane_size_in_bytes[3];
	// Size of the luma plane. Size of chroma is implied by format.
	// Must be same extent as the encoder or decoder.
	int width;
	int height;
	pyrowave_webgpu_cpu_buffer_format format;
} pyrowave_webgpu_cpu_buffer;

// GPU time of the most recent frame, per stage, in nanoseconds. All zero unless the
// device has timestamp queries. Upload and readback are not included: CPU uploads go
// through wgpuQueueWriteTexture / wgpuQueueWriteBuffer and the readback is a copy at
// the end of the last pass.
typedef struct pyrowave_webgpu_timings
{
	uint64_t total_ns;
	// Encoder: DWT, quantize, analyze, resolve, packing.
	// Decoder: dequantize, iDWT.
	uint64_t stage_ns[5];
	int num_stages;
} pyrowave_webgpu_timings;

// Encoder API

typedef struct pyrowave_webgpu_encoder_create_info
{
	pyrowave_webgpu_device device;

	// Luma dimensions. For 420 subsampling both must be even.
	// Both must be in the range [1, 16384], as the bitstream encodes them in 14 bits.
	int width;
	int height;

	pyrowave_webgpu_chroma_subsampling chroma;
} pyrowave_webgpu_encoder_create_info;

typedef struct pyrowave_webgpu_packet
{
	size_t offset;
	size_t size;
} pyrowave_webgpu_packet;

typedef struct pyrowave_webgpu_rate_control
{
	// Very basic, target bitstream for an image must not exceed this size.
	size_t maximum_bitstream_size;
} pyrowave_webgpu_rate_control;

// GPU input: one single channel texture view per plane (r8unorm, r16unorm or any
// float format that can be sampled as float), created with
// WGPUTextureUsage_TextureBinding. Plane 0 is width x height; for 420 chroma, planes 1
// and 2 are (width / 2) x (height / 2), and for 444 they match plane 0.
typedef struct pyrowave_webgpu_gpu_input
{
	WGPUTextureView planes[3];
} pyrowave_webgpu_gpu_input;

PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_create(const pyrowave_webgpu_encoder_create_info *info, pyrowave_webgpu_encoder *encoder);

PYROWAVE_WEBGPU_PUBLIC_API void
pyrowave_webgpu_encoder_destroy(pyrowave_webgpu_encoder encoder);

// Both encode entry points submit the GPU work and return without waiting for it.
// The result is then read back asynchronously. pyrowave_webgpu_encoder_poll()
// reports whether it has arrived; the packet queries below wait for it. Encoding
// again clobbers the previous frame's result. The bitstream carries a small sequence
// counter so the decoder can track frame ordering.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_encode_cpu(pyrowave_webgpu_encoder encoder,
                                   const pyrowave_webgpu_cpu_buffer *input,
                                   const pyrowave_webgpu_rate_control *rate_control);

// The views must stay alive until this returns; WebGPU keeps the textures alive for
// the submitted work.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_encode_gpu(pyrowave_webgpu_encoder encoder,
                                   const pyrowave_webgpu_gpu_input *input,
                                   const pyrowave_webgpu_rate_control *rate_control);

// With wait = false, returns PYROWAVE_WEBGPU_NOT_READY until the last encode's
// result is readable, then PYROWAVE_WEBGPU_SUCCESS. With wait = true, blocks.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_poll(pyrowave_webgpu_encoder encoder, bool wait);

// Only valid after a successful encode, and only for that frame. These wait for the
// GPU as needed. Reports how many packets the frame needs if each may carry at most
// packet_boundary bytes.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_compute_num_packets(pyrowave_webgpu_encoder encoder, size_t packet_boundary,
                                            size_t *num_packets);
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_compute_num_packets_with_padding(
		pyrowave_webgpu_encoder encoder, size_t packet_boundary, size_t padding_size, size_t *num_packets);
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_compute_num_critical_packets(
		pyrowave_webgpu_encoder encoder, int bands, size_t packet_boundary, size_t padding_size, size_t *num_packets);

// `packets` must have room for at least the count reported above; the number
// actually written is returned in out_packets.
//
// packet_boundary is not a hard cap. A coded 32x32 block is the smallest unit a
// packet can carry, so a block larger than packet_boundary on its own becomes one
// oversized packet.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_packetize(pyrowave_webgpu_encoder encoder, pyrowave_webgpu_packet *packets,
                                  size_t packet_boundary, size_t *out_packets,
                                  void *bitstream, size_t size);
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_packetize_with_padding(
		pyrowave_webgpu_encoder encoder, pyrowave_webgpu_packet *packets, size_t packet_boundary, size_t padding_size,
		size_t *out_packets, void *bitstream, size_t size);

// Special purpose for manual packetization. The pointers stay valid until the next
// encode or until the encoder is destroyed.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_get_mapped_raw_bitstream(
		pyrowave_webgpu_encoder encoder, const void **mapped_bitstream, size_t *mapped_bitstream_size,
		const void **mapped_metadata, size_t *mapped_metadata_size);

PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_get_num_active_blocks(pyrowave_webgpu_encoder encoder, int bands, size_t *num_active_blocks);

PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_compute_block_active_words(pyrowave_webgpu_encoder encoder,
		int bands, uint32_t *words, size_t word_count);

PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_encoder_get_timings(pyrowave_webgpu_encoder encoder, pyrowave_webgpu_timings *timings);

// Decoder API

typedef struct pyrowave_webgpu_decoder_create_info
{
	pyrowave_webgpu_device device;

	// Luma dimensions. For 420 subsampling both must be even.
	// Both must be in the range [1, 16384], as the bitstream encodes them in 14 bits.
	int width;
	int height;

	pyrowave_webgpu_chroma_subsampling chroma;
} pyrowave_webgpu_decoder_create_info;

PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_decoder_create(const pyrowave_webgpu_decoder_create_info *info, pyrowave_webgpu_decoder *decoder);

PYROWAVE_WEBGPU_PUBLIC_API void
pyrowave_webgpu_decoder_destroy(pyrowave_webgpu_decoder decoder);

// Throws away all queued packets.
PYROWAVE_WEBGPU_PUBLIC_API void
pyrowave_webgpu_decoder_clear(pyrowave_webgpu_decoder decoder);

// A frame is potentially split into multiple packets.
// If a packet is pushed for a frame that is deemed to arrive earlier, it is dropped.
// A packet pushed for a frame with a higher sequence clears the queued frame and starts a new one.
// Returns PYROWAVE_WEBGPU_ERROR_CORRUPT_BITSTREAM if the data does not parse.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_decoder_push_packet(pyrowave_webgpu_decoder decoder, const void *data, size_t size);

// For error correction purposes, it may be okay to decode a frame which dropped some packets.
PYROWAVE_WEBGPU_PUBLIC_API bool
pyrowave_webgpu_decoder_decode_is_ready(pyrowave_webgpu_decoder decoder, bool allow_partial_frame);

PYROWAVE_WEBGPU_PUBLIC_API bool
pyrowave_webgpu_decoder_decode_is_ready_with_sideband(pyrowave_webgpu_decoder decoder, bool allow_partial_frame,
		int num_pristine_bands, float minimum_packet_ratio,
		const uint32_t *active_block_mask, size_t word_count);

// Decoding may be requested at any time, producing incomplete results if packets are
// missing; missing wavelet coefficients are treated as 0.
//
// Split form: submit uploads the queued packets and submits the decode, poll reports
// whether the readback has arrived (or blocks with wait = true), and read copies the
// planes out. The queued packets can be replaced by the next frame's as soon as
// submit returns.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_decoder_decode_submit(pyrowave_webgpu_decoder decoder);

PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_decoder_decode_poll(pyrowave_webgpu_decoder decoder, bool wait);

// Waits if needed. `output` must be YUV420P or YUV444P, matching the decoder.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_decoder_decode_read(pyrowave_webgpu_decoder decoder, const pyrowave_webgpu_cpu_buffer *output);

// Submit, wait and read in one call.
PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_decoder_decode_cpu(pyrowave_webgpu_decoder decoder, const pyrowave_webgpu_cpu_buffer *output);

PYROWAVE_WEBGPU_PUBLIC_API pyrowave_webgpu_result
pyrowave_webgpu_decoder_get_timings(pyrowave_webgpu_decoder decoder, pyrowave_webgpu_timings *timings);

#ifdef __cplusplus
}
#endif

#endif
