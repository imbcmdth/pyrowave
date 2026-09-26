// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/wavelet_quant.comp. Needs common.wgsl and subgroup.wgsl.
//
// Differences from the GLSL:
// - subgroupClustered{Add,Max}(x, 8) are butterflies of subgroupShuffleXor, which
//   is all core WGSL offers. Like the GLSL, this needs subgroups of at least 8 lanes.
// - WebGPU has no clamp-to-border sampler, so the two textureGatherOffset calls
//   are eight bounds checked textureLoads, which return the same texels.
// - The payload is written a byte at a time from many threads into a u32 array, so
//   bytes are merged into words and ORed in with atomicOr. The host clears the
//   payload buffer before every frame for this; the GLSL only clears its counter.
// - QuantStats {float16_t, uint16_t} is one u32: FP16 bits low, cost high.
// - There is no `precise`; see the notes on FP contraction in webgpu/README.md.
// - All bands are one dispatch (see band_dispatch.wgsl), so the texture is the whole
//   pyramid and the band's level and layer come with its registers.

struct Registers
{
    resolution: vec2<i32>,
    resolution_8x8_blocks: vec2<i32>,
    inv_resolution: vec2<f32>,
    input_layer: f32,
    quant_resolution: f32,
    block_offset: i32,
    block_stride: i32,
    rdo_distortion_scale: f32,
    texture_layer: i32,
    texture_level: i32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};

// struct BlockStats { uint num_planes; QuantStats errors[15]; } is 16 words, and is
// addressed as a flat word array. Out of range errors[] indices then land in the next
// block's words exactly as they do in the GLSL, where WGSL would clamp the index.
const BLOCK_STATS_WORDS: u32 = 16u;

@group(0) @binding(0) var<storage, read> band_registers: array<Registers>;
@group(0) @binding(1) var uTexture: texture_2d_array<f32>; // All levels and layers.
// BlockMeta { code_word, offset }
@group(0) @binding(2) var<storage, read_write> block_meta: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> block_stats: array<u32>;
// Word 0 is the allocation counter, and byte payload starts at byte 8, as in the GLSL.
@group(0) @binding(4) var<storage, read_write> payload_data: array<atomic<u32>>;

const PAYLOAD_DATA_BYTE_OFFSET: u32 = 8u;

var<private> registers: Registers;
var<private> lane: u32;

fn pack_quant_stats(square_error: f32, payload_cost: u32) -> u32
{
    return (pack2x16float(vec2<f32>(square_error, 0.0)) & 0xffffu) | ((payload_cost & 0xffffu) << 16u);
}

fn max4(v: vec4<f32>) -> f32
{
    let v2 = max(v.xy, v.zw);
    return max(v2.x, v2.y);
}

fn clustered_add8_i(v: i32) -> i32
{
    var x = v;
    x += subgroupShuffleXor(x, 1u);
    x += subgroupShuffleXor(x, 2u);
    x += subgroupShuffleXor(x, 4u);
    return x;
}

fn clustered_add8_u(v: u32) -> u32
{
    var x = v;
    x += subgroupShuffleXor(x, 1u);
    x += subgroupShuffleXor(x, 2u);
    x += subgroupShuffleXor(x, 4u);
    return x;
}

fn clustered_add8_f(v: f32) -> f32
{
    var x = v;
    x += subgroupShuffleXor(x, 1u);
    x += subgroupShuffleXor(x, 2u);
    x += subgroupShuffleXor(x, 4u);
    return x;
}

fn clustered_max8_f(v: f32) -> f32
{
    var x = v;
    x = max(x, subgroupShuffleXor(x, 1u));
    x = max(x, subgroupShuffleXor(x, 2u));
    x = max(x, subgroupShuffleXor(x, 4u));
    return x;
}

fn scan_clustered8(v_in: i32) -> i32
{
    var v = v_in;
    for (var i = 1u; i < 8u; i *= 2u)
    {
        let up = subgroupShuffleUp(v, i);
        v += select(0, up, (lane & 7u) >= i);
    }

    return v;
}

struct QuantScale
{
    code: u32,
    scale: f32,
};

fn compute_quant_scale(max_wave_texels: f32) -> QuantScale
{
    var result: QuantScale;
    if (max_wave_texels < 1.0)
    {
        result.code = ENCODE_QUANT_IDENTITY;
        result.scale = 1.0;
    }
    else
    {
        let e = frexp(max_wave_texels - 0.25).exp;
        let target_max = f32(1 << u32(e)) - 0.25;
        let inv_scale = max_wave_texels / target_max;
        result.code = encode_quant_scale(inv_scale);
        result.scale = 1.0 / decode_quant_scale(result.code);
    }
    return result;
}

fn compute_square_error(v_in: mat2x4<f32>, q: i32, num_significant_values: ptr<function, u32>) -> f32
{
    let v = mat2x4<f32>(abs(v_in[0]), abs(v_in[1]));
    var iv = mat2x4<f32>(floor(ldexp(v[0], vec4<i32>(-q))), trunc(ldexp(v[1], vec4<i32>(-q))));
    var count = 0u;
    for (var j = 0; j < 2; j++)
    {
        for (var i = 0; i < 4; i++)
        {
            if (iv[j][i] != 0.0)
            {
                count++;
            }
        }
    }
    iv[0] += select(vec4<f32>(0.0), vec4<f32>(0.5), iv[0] != vec4<f32>(0.0));
    iv[1] += select(vec4<f32>(0.0), vec4<f32>(0.5), iv[1] != vec4<f32>(0.0));
    iv = mat2x4<f32>(trunc(ldexp(iv[0], vec4<i32>(q))), trunc(ldexp(iv[1], vec4<i32>(q))));
    let err0 = v[0] - iv[0];
    let err1 = v[1] - iv[1];
    *num_significant_values = clustered_add8_u(count);
    return (dot(err0, err0) + dot(err1, err1)) * registers.rdo_distortion_scale;
}

struct QuantResult
{
    square_error: f32,
    encode_cost_early: i32,
    block4x2_shifted: i32,
    encode_cost_late_bits: i32,
    quality_planes: i32,
};

fn compute_quant_stats(v: mat2x4<f32>, q: i32, msb_in: i32, block4x2_max_in: i32, inv_quant_squared: f32) -> QuantResult
{
    let block4x2_max = block4x2_max_in >> u32(q);
    var msb = msb_in;

    var wave8_num_significants: u32;
    var result: QuantResult;

    result.square_error = compute_square_error(v, q, &wave8_num_significants) * inv_quant_squared;
    result.block4x2_shifted = block4x2_max;

    result.encode_cost_early = select(0, 1, block4x2_max > 0);
    msb -= q;

    result.quality_planes = 0;

    if (msb >= 3)
    {
        result.quality_planes = msb - 2;
        // Must encode the sign plane if we have quality planes.
        result.encode_cost_early = result.quality_planes + 1;
        result.block4x2_shifted >>= u32(result.quality_planes);
    }

    result.encode_cost_early += firstLeadingBit(result.block4x2_shifted) + 1;
    result.encode_cost_late_bits = 8 * clustered_add8_i(max(result.encode_cost_early - 1, 0)) + i32(wave8_num_significants);
    return result;
}

// Bytes are accumulated into a word and ORed out once the write moves on to the next.
var<private> pending_word: u32;
var<private> pending_word_index: u32;

fn write_payload_byte(byte_offset: u32, value: u32)
{
    let absolute_byte = PAYLOAD_DATA_BYTE_OFFSET + byte_offset;
    let word_index = absolute_byte >> 2u;
    if (word_index != pending_word_index)
    {
        if (pending_word != 0u)
        {
            atomicOr(&payload_data[pending_word_index], pending_word);
        }
        pending_word_index = word_index;
        pending_word = 0u;
    }
    pending_word |= (value & 0xffu) << (8u * (absolute_byte & 3u));
}

fn flush_payload_bytes()
{
    if (pending_word != 0u)
    {
        atomicOr(&payload_data[pending_word_index], pending_word);
    }
    pending_word = 0u;
}

fn plane_byte(abs_quant_texels0: vec4<i32>, abs_quant_texels1: vec4<i32>, q: u32) -> u32
{
    var s0 = vec4<u32>(
        extractBits(u32(abs_quant_texels0.x), q, 1u),
        extractBits(u32(abs_quant_texels0.y), q, 1u),
        extractBits(u32(abs_quant_texels0.z), q, 1u),
        extractBits(u32(abs_quant_texels0.w), q, 1u));
    var s1 = vec4<u32>(
        extractBits(u32(abs_quant_texels1.x), q, 1u),
        extractBits(u32(abs_quant_texels1.y), q, 1u),
        extractBits(u32(abs_quant_texels1.z), q, 1u),
        extractBits(u32(abs_quant_texels1.w), q, 1u));
    s0 <<= vec4<u32>(0u, 1u, 2u, 3u);
    s1 <<= vec4<u32>(4u, 5u, 6u, 7u);
    return s0.x | s0.y | s0.z | s0.w | s1.x | s1.y | s1.z | s1.w;
}

fn encode_payload(block_index_8x8: vec2<i32>, texels_in: mat2x4<f32>)
{
    var texels = texels_in;
    var max_subblock_texel = max(max4(abs(texels[0])), max4(abs(texels[1])));
    var max_wave_texels = clustered_max8_f(max_subblock_texel);
    let qs = compute_quant_scale(max_wave_texels);
    let quant_code = qs.code;
    let quant_scale = qs.scale;
    texels *= quant_scale;
    max_wave_texels *= quant_scale;
    max_subblock_texel *= quant_scale;

    let overall_quant_scale = registers.quant_resolution * quant_scale;
    let inv_quant = 1.0 / overall_quant_scale;
    let inv_quant_squared = inv_quant * inv_quant;
    let abs_quant_texels0 = abs(vec4<i32>(texels[0]));
    let abs_quant_texels1 = abs(vec4<i32>(texels[1]));
    let max_absolute_value = i32(max_wave_texels);
    let block4x2_max = i32(max_subblock_texel);

    let block_index = u32(registers.block_offset + block_index_8x8.y * registers.block_stride + block_index_8x8.x);

    // The entire block quantizes to zero.
    if (max_absolute_value == 0)
    {
        if ((lane & 7u) == 0u)
        {
            block_meta[block_index] = vec2<u32>(0u, 0u);
            block_stats[block_index * BLOCK_STATS_WORDS] = 0u;
            block_stats[block_index * BLOCK_STATS_WORDS + 1u] = pack_quant_stats(0.0, 0u);
        }
        return;
    }

    let msb = firstLeadingBit(max_absolute_value);

    let result = compute_quant_stats(texels, 0, msb, block4x2_max, inv_quant_squared);
    var scan = scan_clustered8(result.encode_cost_early);

    var global_offset = 0u;

    // For feedback, and allocation of payload.
    if ((lane & 7u) == 7u)
    {
        global_offset = atomicAdd(&payload_data[0], u32(scan));
    }
    global_offset = subgroupShuffle(global_offset, lane | 7u);

    scan -= result.encode_cost_early;

    // First, encode the code word.
    let quality_planes = result.quality_planes;
    var code_word = u32(quality_planes) << Q_PLANES_OFFSET;
    code_word = insertBits(code_word, quant_code, QUANT_SCALE_OFFSET, QUANT_SCALE_BITS);
    let plane_code = u32(firstLeadingBit(result.block4x2_shifted) + 1);

    var merged_plane_code = plane_code << ((lane & 7u) * 2u);
    merged_plane_code |= subgroupShuffleXor(merged_plane_code, 1u);
    merged_plane_code |= subgroupShuffleXor(merged_plane_code, 2u);
    merged_plane_code |= subgroupShuffleXor(merged_plane_code, 4u);
    code_word |= merged_plane_code;

    if ((lane & 7u) == 0u)
    {
        block_meta[block_index] = vec2<u32>(code_word, global_offset);
        block_stats[block_index * BLOCK_STATS_WORDS] = u32(msb + 1);
        // We don't care about distortion from 0 quant since we've already made that decision.
        block_stats[block_index * BLOCK_STATS_WORDS + 1u] = pack_quant_stats(0.0, u32(result.encode_cost_late_bits));
    }

    for (var q = 1; q <= msb; q++)
    {
        let quant_result = compute_quant_stats(texels, q, msb, block4x2_max, inv_quant_squared);
        let square_error = clustered_add8_f(quant_result.square_error);

        if ((lane & 7u) == 0u)
        {
            block_stats[block_index * BLOCK_STATS_WORDS + 1u + u32(q)] = pack_quant_stats(
                min(square_error, 60000.0), u32(quant_result.encode_cost_late_bits));
        }
    }

    // Record distortion for throwing away everything.
    let square_error = clustered_add8_f((dot(texels[0], texels[0]) + dot(texels[1], texels[1])) * inv_quant_squared);
    if ((lane & 7u) == 0u)
    {
        block_stats[block_index * BLOCK_STATS_WORDS + 1u + u32(msb + 1)] = pack_quant_stats(min(60000.0, square_error), 0u);
    }

    var byte_offset = u32(scan) + global_offset;
    let need_sign = result.block4x2_shifted != 0 || quality_planes != 0;

    // Don't pack the sign plane until final pass, since we don't know how we quantize yet.
    if (need_sign)
    {
        pending_word = 0u;
        pending_word_index = 0xffffffffu;

        let s0 = vec4<u32>(texels[0] < vec4<f32>(0.0)) << vec4<u32>(0u, 1u, 2u, 3u);
        let s1 = vec4<u32>(texels[1] < vec4<f32>(0.0)) << vec4<u32>(4u, 5u, 6u, 7u);
        let s = s0.x | s0.y | s0.z | s0.w | s1.x | s1.y | s1.z | s1.w;
        write_payload_byte(byte_offset, s);
        byte_offset++;

        let plane_iterations = quality_planes + i32(plane_code);
        var q = plane_iterations - 1;
        loop
        {
            write_payload_byte(byte_offset, plane_byte(abs_quant_texels0, abs_quant_texels1, u32(q)));
            byte_offset++;
            q--;
            if (q < 0)
            {
                break;
            }
        }

        flush_payload_bytes();
    }
}

fn fetch_texel(coord: vec2<i32>) -> f32
{
    // Clamp to border, transparent black.
    if (all(coord < registers.resolution))
    {
        return textureLoad(uTexture, coord, registers.texture_layer, registers.texture_level).x;
    }
    return 0.0;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(num_workgroups) num_workgroups: vec3<u32>,
        @builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
        @builtin(subgroup_size) subgroup_size: u32)
{
    let band = select_band(workgroup_id, num_workgroups);
    if (!band.valid)
    {
        return;
    }
    // A let of a read-only storage load at a uniform index is uniform for WGSL's
    // uniformity analysis; the module scope copy for helper functions is not.
    let regs = band_registers[band.band];
    registers = regs;
    let wg_id = band.id;

    let subgroup_id = allocate_subgroup_id(local_invocation_index, subgroup_invocation_id);
    let local_index = subgroup_id * subgroup_size + subgroup_invocation_id;
    lane = subgroup_invocation_id;

    let block_local_index = extractBits(local_index, 0u, 3u);
    let block_x = extractBits(local_index, 3u, 2u);
    let block_y = extractBits(local_index, 5u, 2u);

    // Each thread individually encodes 8 values.
    let local_coord = unswizzle8x8(block_local_index << 3u);

    var coord = vec2<i32>(wg_id) * 32;
    coord += 8 * vec2<i32>(i32(block_x), i32(block_y));
    coord += local_coord;

    let block_index = 4 * vec2<i32>(wg_id) + vec2<i32>(i32(block_x), i32(block_y));

    // textureGatherOffset(uTexture, uv, ivec2(1, 1)).wxzy and ivec2(3, 1).
    let texels0 = vec4<f32>(
        fetch_texel(coord + vec2<i32>(0, 0)), fetch_texel(coord + vec2<i32>(0, 1)),
        fetch_texel(coord + vec2<i32>(1, 0)), fetch_texel(coord + vec2<i32>(1, 1)));
    let texels1 = vec4<f32>(
        fetch_texel(coord + vec2<i32>(2, 0)), fetch_texel(coord + vec2<i32>(2, 1)),
        fetch_texel(coord + vec2<i32>(3, 0)), fetch_texel(coord + vec2<i32>(3, 1)));
    let scaled_texels0 = texels0 * regs.quant_resolution;
    let scaled_texels1 = texels1 * regs.quant_resolution;
    let in_bounds = all(block_index < regs.resolution_8x8_blocks);
    if (in_bounds)
    {
        encode_payload(block_index, mat2x4<f32>(scaled_texels0, scaled_texels1));
    }
}
