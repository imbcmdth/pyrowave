This is a port of the PyroWave encoder and decoder to WebGPU, written against the
standard `webgpu.h` from [webgpu-headers](https://github.com/webgpu-native/webgpu-headers).
It has no Granite or Vulkan dependency, so the same code can run on wgpu-native, on
Dawn, or on a `webgpu.h` shim over another WebGPU implementation.

The bitstream is the one in `../bitstream/bitstream.md`. Streams from the Vulkan
encoder decode here, and streams encoded here decode with the Vulkan decoder.

This is covered by the standard MIT license available in ../LICENSE.

## Layout

* `pyrowave_webgpu.h`: C API. Everything is prefixed `pyrowave_webgpu_`, so it can be
  used next to `pyrowave.h` in one program. The shape follows `../metal/pyrowave_metal.h`.
* `pyrowave_webgpu_*.cpp`: device, encoder and decoder. The Granite-free bitstream
  code (block layout, packet parsing, packetization) is `../metal/pyrowave_bitstream.cpp`,
  compiled in as is.
* `shaders/*.wgsl`: hand ports of `../shaders/*.comp`. They are embedded into the
  library at build time, so nothing is read from disk at run time.
* `tools/`: command line tools, see below.

## Building

wgpu-native (prebuilt, pinned):

```
cmake -P webgpu/fetch_wgpu_native.cmake
cmake -S . -B build -DPYROWAVE_WEBGPU=ON
```

`PYROWAVE_WEBGPU` is off by default; with it off the build is unchanged. The
`webgpu/` directory also builds on its own, without Granite:

```
cmake -S webgpu -B build-webgpu
```

Dawn instead of wgpu-native: pass `-DPYROWAVE_WEBGPU_DAWN_DIR=<Dawn install>`, the
directory with `lib/cmake/Dawn`. Any other implementation works by defining a
`webgpu` CMake target before adding this directory.

Implementation notes, as of wgpu-native v29.0.1.1 and Dawn nightly 20260925:

* wgpu-native reports subgroups only as its own `WGPUNativeFeature_Subgroup`, not the
  standard feature, and its WGSL front end rejects `enable subgroups;`. The build
  defines `PYROWAVE_WEBGPU_WGPU_NATIVE` for it, which enables the one fallback in
  `pyrowave_webgpu_common.cpp`. Its D3D12 backend has no subgroups at all, so it runs
  on Vulkan. `wgpuHasInstanceFeature` is not implemented, and workgroup memory was
  not zero initialized, which the shaders no longer rely on.
* Dawn's prebuilt Windows libraries need MSVC 14.5 or newer to link, and
  `vulkan-1.dll` (and for D3D12 `dxcompiler.dll` and `dxil.dll`) next to the
  executable. Dawn rounds timestamps to about 65 microseconds.

## Requirements

* The `subgroups` feature. Only standard WGSL subgroup builtins are used: ballot,
  broadcast, shuffle, shuffle xor, shuffle up and inclusive add.
* Subgroup size: WebGPU cannot request one, so the shaders do not assume one.
  Where the GLSL uses clustered operations or relies on a size forced with subgroup
  size control, the WGSL uses shuffle butterflies inside clusters and moves scans that
  cross subgroups into workgroup memory. The encoder needs subgroups of 16 to 64
  lanes, the range the Vulkan encoder asks for; the decoder takes 4 to 128. The
  adapter's reported range is checked when it reports one (wgpu-native v29 reports
  0 to 0, meaning unknown).
* Default limits, apart from `maxStorageBufferBindingSize` and `maxBufferSize`, which
  are raised to the adapter's limits for large frames. Block packing uses exactly the
  default 8 storage buffers per stage.
* `timestamp-query` is optional, for per stage GPU timings.
* `shader-f16` is not used. `PYROWAVE_PRECISION` 1 (the default) and 2 behave as in
  the Vulkan build; 0 (FP16 arithmetic) falls back to 1.

## How the WGSL differs from the GLSL

Each shader lists its differences at the top. In short:

* No `subgroup_id` builtin in core WGSL, and no guarantee of how
  `local_invocation_index` maps onto subgroups. Each subgroup takes a ticket from a
  workgroup counter instead, which gives the same subgroup-major indexing.
* No 8 or 16-bit storage. Bytes and halves are read out of u32 words, and the
  encoder writes them with `atomicOr` into words it cleared first.
* No clamp-to-border sampler: the quantizer uses bounds checked `textureLoad`.
* The wavelet pyramid is `r32float`, the only single channel float format core
  WebGPU can write from a shader. Precision 1 rounds the two highest resolution
  levels to FP16 to match the Vulkan build's R16F.
* The GLSL transforms fetch texels through `mediump` samplers, and NVIDIA's Vulkan
  driver does return FP16 texels for those. The WGSL rounds the fetched texels to
  FP16 explicitly, so the result is the same everywhere.
* FP16 rounding is done with integer math. `unpack2x16float(pack2x16float(x))` was
  folded away by the NVIDIA driver, `quantizeToF16` flushes denormals, and
  `pack2x16float` does not round to nearest on D3D12.
* WebGPU implementations put a barrier between dispatches that write storage.
  Upstream records one dispatch per band, about 40 per stage, which made quantize and
  packing 10 to 20 times slower here. Quantize, analyze, packing and each dequant
  level run all their bands in one dispatch instead (`shaders/band_dispatch.wgsl`).
* `analyze_rate_control_finalize` runs 256 invocations playing the GLSL's 512, since
  256 is the default limit. Its scan stops at `step < 256` as in the GLSL, so each
  entry sums a window of 256 invocations rather than a full prefix. That is kept so
  both backends make the same rate control decisions, but it looks unintended.
* The encoder's bitstream bytes are deterministic: unused bits at the end of a
  packet are zero. The Vulkan encoder leaves whatever was in memory there.

## Tools

* `pyrowave-webgpu-encode <in.y4m> <out.pyrowave> <bytes_per_frame> [--frames N] [--timestamps] [--transfer-bench] [--gpu-input | --nv12]`
* `pyrowave-webgpu-decode <in.pyrowave> <out.y4m> [--timestamps]`
* `pyrowave-vulkan-cli encode|decode ...`: the same through the Vulkan C API, built
  with `PYROWAVE_WEBGPU=ON` from the top level.
* `pyrowave-vulkan-dwt-dump dwt|idwt ...`: writes the Vulkan encoder's wavelet
  pyramid, or the decoder's output before conversion to 8 bits, as float32.
  `PYROWAVE_WEBGPU_DUMP_PYRAMID=<file>` makes the WebGPU encoder and decoder write
  their pyramid in the same layout.

The file format is the one `../encode.cpp` and `../decode.cpp` use.

Environment variables: `PYROWAVE_PRECISION`, `PYROWAVE_WEBGPU_BACKEND` (vulkan,
d3d12, metal), `PYROWAVE_WEBGPU_TIMED_WAIT_ANY=1`.

## Validation

On an RTX 4090 (driver 610.188), wgpu-native v29.0.1.1 on Vulkan, against the
Vulkan build at upstream 89f7e47, 8-bit y4m input:

* Encoder: the forward transform is bit identical to the Vulkan one at precision 1
  and 2. The streams have the same size frame for frame, the Vulkan decoder turns
  them into exactly the same pictures as the Vulkan encoder's streams, and the coded
  blocks compare equal one by one on the frames checked. Tested at 1920x1080,
  1366x768, 1280x720 4:2:0 and 4:4:4, 642x362 and 34x30. Dawn on Vulkan produces the
  same bytes as wgpu-native. Dawn on D3D12 differs in a coefficient or two in a few
  percent of blocks (a different shader compiler), at the same PSNR.
* Decoder: the 8-bit output is the Vulkan decoder's float output rounded to nearest,
  bit for bit (checked on a 1080p frame). The Vulkan decoder's own 8-bit output
  differs by 1 in 0.05 to 3 % of samples depending on content, because NVIDIA's
  R8_UNORM image store rounds as `(floor(x * 4096) * 255 + 2047) >> 12` rather than to
  nearest. Never by more than 1.

PSNR against the source with the ffmpeg psnr filter, 90 frames of 1920x1080 4:2:0,
each backend encoding and decoding its own stream:

| clip | bytes per frame | Vulkan Y / U / V | WebGPU Y / U / V |
|---|---|---|---|
| real footage | 60000 | 34.608 / 41.370 / 36.997 | 34.608 / 41.444 / 36.985 |
| real footage | 120000 | 38.175 / 43.841 / 41.712 | 38.174 / 43.864 / 41.687 |
| real footage | 250000 | 44.089 / 50.043 / 48.999 | 44.090 / 50.061 / 49.005 |
| testsrc2 | 60000 | 36.377 / 36.047 / 35.305 | 36.377 / 36.047 / 35.305 |
| testsrc2 | 120000 | 39.975 / 42.722 / 44.740 | 39.976 / 42.723 / 44.733 |
| testsrc2 | 250000 | 52.496 / 57.230 / 57.120 | 52.493 / 57.239 / 57.110 |

Per frame at 1920x1080, 250000 bytes per frame, one frame at a time through CPU
memory, mean over 89 frames:

| | Vulkan (C API) | WebGPU (wgpu-native) |
|---|---|---|
| encode, wall clock, upload to packets | 0.74 ms | 0.63 ms |
| encode, GPU: DWT / quant / analyze / resolve / packing | 0.023 / 0.047 / 0.016 / 0.005 / 0.022 ms | 0.081 / 0.053 / 0.015 / 0.005 / 0.034 ms |
| decode, wall clock, packets to planes | 0.51 ms | 0.50 ms |
| decode, GPU: dequant / iDWT | 0.032 / 0.024 ms | 0.055 / 0.051 ms |

Uploading one 4:2:0 frame on its own takes 0.12 ms and reading back 250000 bytes
0.18 ms (round trips, wall clock), so at this size moving frames through CPU memory
costs more than the GPU work. The remaining GPU gap is mostly the transforms: they
still run one dispatch per component and level, with barriers between them, and the
pyramid is twice the size of the Vulkan build's R16F levels.

## Notes for a WASM build

* No threads, exceptions or run time file access. `getenv` is read for the variables
  above, and `fopen` only for `PYROWAVE_WEBGPU_DUMP_PYRAMID`.
* Waiting is in one place, `pyrowave_webgpu_device_opaque::wait()`: it blocks in
  `wgpuInstanceWaitAny` if `timed_wait_any` is set, and otherwise spins on
  `wgpuInstanceProcessEvents`. Device creation and pipeline creation wait there. Per
  frame, `pyrowave_webgpu_encoder_poll()` and `pyrowave_webgpu_decoder_decode_poll()`
  with `wait = false` let a host drive the readback without blocking; the packet
  queries and `decode_read` wait if it has not landed.
* Readback uses `wgpuBufferMapAsync` and `wgpuBufferGetConstMappedRange`, once per
  frame, about 260 KB for the encoder at this rate and 3 MB for a 1080p decode.
* If the WGSL front end rejects `enable subgroups;`, as naga does, the shaders are
  compiled again without it.
