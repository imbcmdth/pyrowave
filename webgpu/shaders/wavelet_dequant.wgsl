// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/wavelet_dequant.comp (the STORAGE_MODE 0 path).
// Needs common.wgsl and subgroup.wgsl.
//
// Differences from the GLSL:
// - The payload is a u32 array; 8 and 16-bit loads extract from it.
// - The two prefix sums that cross subgroup boundaries (the byte offsets of the 16
//   8x8 blocks, and the per subgroup sign bit totals) go through workgroup memory
//   instead of subgroup scans, so the result does not depend on the subgroup size,
//   which WebGPU does not let us choose. The in-subgroup sign scan still uses
//   subgroupInclusiveAdd.
// - Output is r32float, rounded through FP16 where the GLSL stores R16F.
// - All bands of a level are one dispatch (see band_dispatch.wgsl), writing a view of
//   all 12 layers of that level; output_layer is component * 4 + band.

struct Registers
{
    resolution: vec2<i32>,
    output_layer: i32,
    block_offset_32x32: i32,
    block_stride_32x32: i32,
    store_fp16: u32,
    padding0: u32,
    padding1: u32,
};

@group(0) @binding(0) var<storage, read> band_registers: array<Registers>;
@group(0) @binding(1) var uDequantImg: texture_storage_2d_array<r32float, write>;
@group(0) @binding(2) var<storage, read> payload_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> payload_data: array<u32>;

const WORKGROUP_SIZE: u32 = 128u;

var<private> registers: Registers;

fn read_payload_u8(index: u32) -> u32
{
    return extract_u8(payload_data[index >> 2u], index);
}

fn read_payload_u16(index: u32) -> u32
{
    return extract_u16(payload_data[index >> 1u], index);
}

fn decode_payload(code_word: u32, q_bits: u32, offset: u32, block_index: u32) -> mat2x4<f32>
{
    let empty_block = code_word == 0u;
    if (empty_block)
    {
        return mat2x4<f32>();
    }

    let bit_offset = 2u * block_index;

    // First, we need to compute the offset that our 4x2 block starts on.
    let lsbs = code_word & 0x5555u;
    var msbs = code_word & 0xaaaau;
    let msbs_shift = msbs >> 1u;
    msbs |= msbs_shift;

    var byte_offset =
        countOneBits(extractBits(lsbs, 0u, bit_offset)) +
        countOneBits(extractBits(msbs, 0u, bit_offset)) +
        q_bits * block_index + offset;

    // Eagerly load the data to keep latency down.
    var payload = read_payload_u8(byte_offset);

    let local_control_word = extractBits(code_word, bit_offset, 2u);
    var decoded_abs = array<i32, 8>(0, 0, 0, 0, 0, 0, 0, 0);
    let plane_iterations = i32(q_bits + local_control_word);

    for (var q = plane_iterations - 1; q >= 0; q--)
    {
        for (var b = 0u; b < 8u; b++)
        {
            let decoded = i32(extractBits(payload, b, 1u));
            decoded_abs[b] = insertBits(decoded_abs[b], decoded, u32(q), 1u);
        }
        byte_offset++;
        payload = read_payload_u8(byte_offset);
    }

    var m: mat2x4<f32>;

    for (var i = 0; i < 4; i++)
    {
        for (var j = 0; j < 2; j++)
        {
            var v = f32(decoded_abs[i * 2 + j]);
            if (v != 0.0)
            {
                v += 0.5;
            }
            m[j][i] = v;
        }
    }

    return m;
}

var<workgroup> shared_sign_offset: u32;
var<workgroup> shared_byte_cost: array<u32, 16>;
var<workgroup> shared_plane_byte_offsets: array<u32, 16>;
var<workgroup> shared_sign_scan: array<u32, WORKGROUP_SIZE / 4u>;

const MaxScaleExp: i32 = 4;

fn decode_quant(quant_code: u32) -> f32
{
    // Custom FP formulation for numbers in (0, 16) range.
    let e = MaxScaleExp - i32(quant_code >> 3u);
    let m = i32(quant_code) & 0x7;
    let inv_quant = (1.0 / (8.0 * 1024.0 * 1024.0)) * f32((8 + m) * (1 << u32(20 + e)));
    return inv_quant;
}

@compute @workgroup_size(WORKGROUP_SIZE)
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
    let num_subgroups = max(WORKGROUP_SIZE / subgroup_size, 1u);

    let fp16 = regs.store_fp16 != 0u;

    let block_index_32x32 = u32(regs.block_offset_32x32 +
        i32(wg_id.y) * regs.block_stride_32x32 +
        i32(wg_id.x));

    let block_local_index = extractBits(local_index, 0u, 3u);
    let block_x = extractBits(local_index, 3u, 2u);
    let block_y = extractBits(local_index, 5u, 2u);
    let linear_block = block_y * 4u + block_x;

    // Each thread individually decodes 8 values.
    let local_coord = unswizzle8x8(block_local_index << 3u);

    var coord = vec2<i32>(wg_id) * 32;
    coord += 8 * vec2<i32>(i32(block_x), i32(block_y));
    coord += local_coord;

    let offset_u32 = payload_offsets[block_index_32x32];

    if (offset_u32 == 0xffffffffu)
    {
        for (var j = 0; j < 2; j++)
        {
            for (var i = 0; i < 4; i++)
            {
                textureStore(uDequantImg, coord + vec2<i32>(i, j), regs.output_layer, vec4<f32>(0.0));
            }
        }
        return;
    }

    let ballot = payload_data[offset_u32] & 0xffffu;
    let q_code = payload_data[offset_u32 + 1u] & 0xffu;

    if (local_index < 16u)
    {
        var control_word = 0u;
        var q_bits = 0u;

        if (extractBits(ballot, local_index, 1u) != 0u)
        {
            let local_code_offset = countOneBits(extractBits(ballot, 0u, local_index));
            control_word = read_payload_u16(offset_u32 * 2u + 4u + local_code_offset);
            q_bits = read_payload_u8(offset_u32 * 4u + 8u + countOneBits(ballot) * 2u + local_code_offset) & 0xfu;
        }

        let lsbs = control_word & 0x5555u;
        var msbs = control_word & 0xaaaau;
        let msbs_shift = msbs >> 1u;
        msbs |= msbs_shift;
        shared_byte_cost[local_index] = countOneBits(lsbs) + countOneBits(msbs) + q_bits * 8u;
    }

    workgroupBarrier();

    if (local_index < 16u)
    {
        // Exclusive prefix sum over the 16 8x8 blocks. The GLSL does this with a
        // subgroupInclusiveAdd across lanes 0-15, which needs 16 wide subgroups.
        var byte_scan = offset_u32 * 4u + 8u + 3u * countOneBits(ballot);
        for (var i = 0u; i < local_index; i++)
        {
            byte_scan += shared_byte_cost[i];
        }
        shared_plane_byte_offsets[local_index] = byte_scan;
        if (local_index == 15u)
        {
            shared_sign_offset = 8u * (byte_scan + shared_byte_cost[15]);
        }
    }

    workgroupBarrier();

    var v: mat2x4<f32>;
    var significant_count: u32;

    if (extractBits(ballot, linear_block, 1u) != 0u)
    {
        let local_code_offset = countOneBits(extractBits(ballot, 0u, linear_block));

        let control_word = read_payload_u16(offset_u32 * 2u + 4u + local_code_offset);
        let control_word2 = read_payload_u8(offset_u32 * 4u + 8u + countOneBits(ballot) * 2u + local_code_offset);

        v = decode_payload(control_word, control_word2 & 0xfu,
            shared_plane_byte_offsets[linear_block], block_local_index);

        significant_count = 0u;
        for (var j = 0; j < 2; j++)
        {
            for (var i = 0; i < 4; i++)
            {
                significant_count += u32(v[j][i] != 0.0);
            }
        }

        let q = decode_quant(q_code);
        let inv_scale = q * decode_quant_scale(extractBits(control_word2, QUANT_SCALE_OFFSET - 16u, QUANT_SCALE_BITS));

        v *= inv_scale;
    }
    else
    {
        v = mat2x4<f32>();
        significant_count = 0u;
    }

    // Figure out how many significant coefficients we have.
    let significant_scan = subgroupInclusiveAdd(significant_count);
    if (subgroup_invocation_id == subgroup_size - 1u)
    {
        shared_sign_scan[subgroup_id] = significant_scan;
    }

    workgroupBarrier();

    // Scan the per subgroup totals. There are at most WORKGROUP_SIZE / 4 of them and
    // typically 4, so one thread doing it serially is cheaper than it sounds.
    if (local_index == 0u)
    {
        var total = 0u;
        for (var i = 0u; i < num_subgroups; i++)
        {
            total += shared_sign_scan[i];
            shared_sign_scan[i] = total;
        }
    }

    workgroupBarrier();

    // Compute where we need to start reading sign bits from.
    var sign_offset = shared_sign_offset + significant_scan - significant_count;
    if (subgroup_id != 0u)
    {
        sign_offset += shared_sign_scan[subgroup_id - 1u];
    }

    // Read out all sign bits we could possibly access per thread.
    var sign_word = payload_data[sign_offset / 32u + 0u];
    let sign_word_upper = payload_data[sign_offset / 32u + 1u];

    let masked_sign_offset = sign_offset & 31u;
    if (masked_sign_offset != 0u)
    {
        sign_word >>= masked_sign_offset;
        sign_word |= sign_word_upper << (32u - masked_sign_offset);
    }

    var sign_counter = 0u;

    // Clock out the sign bits as needed.
    for (var i = 0; i < 4; i++)
    {
        for (var j = 0; j < 2; j++)
        {
            if (v[j][i] != 0.0)
            {
                v[j][i] *= 1.0 - 2.0 * f32(extractBits(sign_word, sign_counter, 1u));
                sign_counter++;
            }
        }
    }

    // Write output.
    for (var j = 0; j < 2; j++)
    {
        for (var i = 0; i < 4; i++)
        {
            textureStore(uDequantImg, coord + vec2<i32>(i, j), regs.output_layer,
                         vec4<f32>(round_wavelet(v[j][i], fp16)));
        }
    }
}
