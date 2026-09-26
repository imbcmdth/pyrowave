// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/constants.h, shaders/dwt_quant_scale.h and shaders/dwt_swizzle.h.
// WGSL has no #include, so the host concatenates this file in front of every shader
// that needs it (see shader_source() in pyrowave_webgpu_common.cpp).

// constants.h
const Q_PLANES_OFFSET: u32 = 16u;
const Q_PLANES_BITS: u32 = 4u;

const QUANT_SCALE_OFFSET: u32 = 20u;
const QUANT_SCALE_BITS: u32 = 4u;

// dwt_quant_scale.h
fn decode_quant_scale(code: u32) -> f32
{
    // Minimum scale: 0.25
    // Maximum scale: ~2.2
    return f32(code) / 8.0 + 0.25;
}

const ENCODE_QUANT_IDENTITY: u32 = 6u;

fn encode_quant_scale(scale: f32) -> u32
{
    // Round the quant scale FP up so that the quantizer scale effectively rounds down.
    return u32(ceil((scale - 0.25) * 8.0));
}

// dwt_swizzle.h
fn unswizzle4x8(index: u32) -> vec2<i32>
{
    var y = extractBits(index, 0u, 1u);
    let x = extractBits(index, 1u, 2u);
    y |= extractBits(index, 3u, 2u) << 1u;
    return vec2<i32>(i32(x), i32(y));
}

fn unswizzle8x8(index: u32) -> vec2<i32>
{
    var y = extractBits(index, 0u, 1u);
    var x = extractBits(index, 1u, 2u);
    y |= extractBits(index, 3u, 2u) << 1u;
    x |= extractBits(index, 5u, 1u) << 2u;
    return vec2<i32>(i32(x), i32(y));
}

// Reads a byte or a 16-bit word out of a u32 array. WGSL has no 8 or 16-bit
// storage types, so the GLSL uint8_t / uint16_t buffer aliases turn into these.
fn extract_u8(word: u32, byte_index: u32) -> u32
{
    return extractBits(word, 8u * (byte_index & 3u), 8u);
}

fn extract_u16(word: u32, half_index: u32) -> u32
{
    return extractBits(word, 16u * (half_index & 1u), 16u);
}

// Rounds to the nearest FP16 value (ties to even), keeping FP16 denormals, as a
// conversion to FP16 and back would. Done with integer math because
// unpack2x16float(pack2x16float(x)) can be folded away by the driver, and
// quantizeToF16 flushes FP16 denormals and was measurably slower.
// Values beyond the FP16 range are not handled; wavelet coefficients stay far below.
fn round_to_f16_vec4(v: vec4<f32>) -> vec4<f32>
{
    let bits = bitcast<vec4<u32>>(v);
    let normal = bitcast<vec4<f32>>((bits + 0xfffu + ((bits >> vec4<u32>(13u)) & vec4<u32>(1u))) & vec4<u32>(0xffffe000u));
    let denormal = round(v * 16777216.0) * (1.0 / 16777216.0);
    return select(normal, denormal, abs(v) < vec4<f32>(6.103515625e-05));
}

fn round_to_f16(v: f32) -> f32
{
    return round_to_f16_vec4(vec4<f32>(v)).x;
}

// The wavelet pyramid is always an r32float texture, since that is the only single
// channel float format WebGPU can write from a shader without optional features. The
// GLSL stores the two highest resolution levels as R16F under PRECISION 1, so round
// through FP16 there to get the same values.
fn round_wavelet(v: f32, fp16: bool) -> f32
{
    if (fp16)
    {
        return round_to_f16(v);
    }
    return v;
}

// The GLSL transforms read through `mediump` samplers. That is only a hint, but
// NVIDIA's Vulkan driver does return FP16 texels for it, which is what the Vulkan
// encoder output was checked against, so the WGSL does the same rounding explicitly.
fn mediump_texels(v: vec4<f32>) -> vec4<f32>
{
    return round_to_f16_vec4(v);
}
