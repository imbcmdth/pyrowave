// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/block_packing.comp. Needs common.wgsl and subgroup.wgsl.
//
// Differences from the GLSL:
// - The GLSL writes the bitstream through uint8_t / uint16_t aliases of the output
//   buffer. WGSL has neither, so the output is an atomic u32 array: each 16 lane
//   cluster first zeroes the words of the packet it allocated, and after a
//   storageBarrier every byte and 16-bit field is ORed in. Bytes a thread writes in
//   sequence are merged into one atomicOr per word.
// - Unused bits at the end of a packet are therefore always zero. In the GLSL they are
//   whatever the buffer or shared_sign_bank held before, so the two encoders do not
//   produce byte identical bitstreams even when they code identical data. Compare
//   decoded output instead.
// - subgroupClusteredAdd(x, 16) is a subgroupShuffleXor butterfly, and the ballot of
//   a 16 lane cluster is picked out of the subgroup ballot by lane, which covers any
//   subgroup size from 16 to 128. The subgroupBarrier becomes a workgroupBarrier.
// - All bands are one dispatch, see band_dispatch.wgsl.

struct Registers
{
    resolution: vec2<i32>,
    resolution_32x32_blocks: vec2<i32>,
    resolution_8x8_blocks: vec2<i32>,
    quant_resolution_code: u32,
    sequence_code: u32,
    block_offset_32x32: i32,
    block_stride_32x32: i32,
    block_offset_8x8: i32,
    block_stride_8x8: i32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

// struct BlockStats { uint num_planes; QuantStats errors[15]; }, as a flat word array.
const BLOCK_STATS_WORDS: u32 = 16u;

@group(0) @binding(0) var<storage, read> band_registers: array<Registers>;
@group(0) @binding(1) var<storage, read_write> bitstream_data: array<atomic<u32>>;
// BitstreamPacket { offset, num_words }
@group(0) @binding(2) var<storage, read_write> bitstream_meta: array<vec2<u32>>;
// BlockMeta { code_word, offset }
@group(0) @binding(3) var<storage, read> block_meta: array<vec2<u32>>;
// Word 1 is the bitstream allocation counter, and byte payload starts at byte 8.
@group(0) @binding(4) var<storage, read_write> payload_data: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> block_stats: array<u32>;
@group(0) @binding(6) var<storage, read> quant_data: array<i32>;

const PAYLOAD_DATA_BYTE_OFFSET: u32 = 8u;
const BITSTREAM_PAYLOAD_COUNTER: u32 = 1u;

var<private> registers: Registers;
var<private> lane: u32;

fn compute_required_8x8_size(control_word: u32) -> u32
{
    let q_bits = extractBits(control_word, Q_PLANES_OFFSET, Q_PLANES_BITS);
    let lsbs = control_word & 0x5555u;
    var msbs = control_word & 0xaaaau;
    let msbs_shift = msbs >> 1u;
    msbs |= msbs_shift;
    return countOneBits(lsbs) + countOneBits(msbs) + q_bits * 8u;
}

fn quantize_code_word(control_word_in: u32, quant_in: i32) -> u32
{
    var control_word = control_word_in;
    var quant = quant_in;

    if (quant != 0 && control_word != 0u)
    {
        var q_bits = i32(extractBits(control_word, Q_PLANES_OFFSET, Q_PLANES_BITS));
        let sub_quant = min(q_bits, quant);
        q_bits -= sub_quant;
        quant -= sub_quant;

        if (quant != 0)
        {
            quant = min(quant, 3);

            var plane0 = control_word & 0x5555u;
            var plane1 = (control_word & 0xaaaau) >> 1u;
            var plane2 = plane0 & plane1;

            loop
            {
                plane0 = plane1;
                plane1 = plane2;
                plane2 = 0u;
                quant--;
                if (quant == 0)
                {
                    break;
                }
            }

            plane0 &= ~plane1;

            let new_control_word = plane0 | (plane1 << 1u);
            control_word = insertBits(control_word, new_control_word, 0u, 16u);
        }

        control_word = insertBits(control_word, u32(q_bits), Q_PLANES_OFFSET, Q_PLANES_BITS);
    }

    return control_word;
}

// Reading bytes of the quantizer's payload, one word cached.
var<private> input_word: u32;
var<private> input_word_index: u32 = 0xffffffffu;

fn read_payload_byte(byte_offset: u32) -> u32
{
    let absolute_byte = PAYLOAD_DATA_BYTE_OFFSET + byte_offset;
    let word_index = absolute_byte >> 2u;
    if (word_index != input_word_index)
    {
        input_word = atomicLoad(&payload_data[word_index]);
        input_word_index = word_index;
    }
    return extract_u8(input_word, absolute_byte);
}

// Writing bytes of the bitstream, merged into one atomicOr per word.
var<private> output_word: u32;
var<private> output_word_index: u32 = 0xffffffffu;

fn write_bitstream_byte(byte_offset: u32, value: u32)
{
    let word_index = byte_offset >> 2u;
    if (word_index != output_word_index)
    {
        if (output_word != 0u)
        {
            atomicOr(&bitstream_data[output_word_index], output_word);
        }
        output_word_index = word_index;
        output_word = 0u;
    }
    output_word |= (value & 0xffu) << (8u * (byte_offset & 3u));
}

fn flush_bitstream_bytes()
{
    if (output_word != 0u)
    {
        atomicOr(&bitstream_data[output_word_index], output_word);
    }
    output_word = 0u;
    output_word_index = 0xffffffffu;
}

// ORs the low num_bytes bytes of value in at byte_offset, which need not be aligned.
fn or_bitstream_bytes(byte_offset: u32, value: u32, num_bytes: u32)
{
    let shift = 8u * (byte_offset & 3u);
    let word_index = byte_offset >> 2u;
    var v = value;
    if (num_bytes < 4u)
    {
        v &= (1u << (8u * num_bytes)) - 1u;
    }

    if (v == 0u)
    {
        return;
    }

    atomicOr(&bitstream_data[word_index], v << shift);
    if (shift != 0u)
    {
        let upper = v >> (32u - shift);
        if (upper != 0u)
        {
            atomicOr(&bitstream_data[word_index + 1u], upper);
        }
    }
}

fn copy_bytes(output_offset: ptr<function, u32>, input_offset_in: u32, count_in: u32) -> u32
{
    var significant_mask = 0u;
    var input_offset = input_offset_in;
    var count = count_in;

    loop
    {
        let in_data = read_payload_byte(input_offset);
        // If we observe any 1 in the non-sign planes, it's not deadzone quantized.
        significant_mask |= in_data;
        write_bitstream_byte(*output_offset, in_data);
        *output_offset += 1u;
        count--;
        input_offset++;
        if (count == 0u)
        {
            break;
        }
    }

    return significant_mask;
}

fn modify_quant_code(code: u32, quant: i32) -> u32
{
    var e = i32(extractBits(code, 3u, 5u));
    e = max(e - quant, 0);
    return insertBits(code, u32(e), 3u, 5u);
}

fn inclusive_add_clustered16(v_in: u32) -> u32
{
    var v = v_in;
    for (var i = 1u; i < 16u; i *= 2u)
    {
        let up = subgroupShuffleUp(v, i);
        v += select(0u, up, (lane & 15u) >= i);
    }

    return v;
}

fn clustered_add16(v_in: u32) -> u32
{
    var v = v_in;
    v += subgroupShuffleXor(v, 1u);
    v += subgroupShuffleXor(v, 2u);
    v += subgroupShuffleXor(v, 4u);
    v += subgroupShuffleXor(v, 8u);
    return v;
}

var<workgroup> shared_sign_bank: array<array<atomic<u32>, 1024 / 32>, 4>;
var<private> pending_sign_write: u32 = 0u;
var<private> pending_sign_mask: u32 = 0u;

fn append_sign_plane(bank: u32, local_sign_offset: ptr<function, u32>, sign_mask: u32, significant_mask_in: u32)
{
    var significant_mask = significant_mask_in;

    // Clock out one bit a time. This seems kinda slow.
    while (significant_mask != 0u)
    {
        let bit = firstTrailingBit(significant_mask);
        significant_mask &= significant_mask - 1u;
        let out_bit = *local_sign_offset & 31u;
        pending_sign_write = insertBits(pending_sign_write, extractBits(sign_mask, bit, 1u), out_bit, 1u);
        pending_sign_mask = insertBits(pending_sign_mask, 1u, out_bit, 1u);

        if (out_bit == 31u)
        {
            if (pending_sign_mask == 0xffffffffu)
            {
                atomicStore(&shared_sign_bank[bank][*local_sign_offset / 32u], pending_sign_write);
            }
            else
            {
                atomicAnd(&shared_sign_bank[bank][*local_sign_offset / 32u], ~pending_sign_mask);
                atomicOr(&shared_sign_bank[bank][*local_sign_offset / 32u], pending_sign_write & pending_sign_mask);
            }

            pending_sign_mask = 0u;
        }

        *local_sign_offset += 1u;
    }
}

fn flush_sign_plane(bank: u32, local_sign_offset: u32)
{
    if (pending_sign_mask != 0u)
    {
        atomicAnd(&shared_sign_bank[bank][local_sign_offset / 32u], ~pending_sign_mask);
        atomicOr(&shared_sign_bank[bank][local_sign_offset / 32u], pending_sign_write & pending_sign_mask);
        pending_sign_mask = 0u;
    }
}

const HeaderSize: u32 = 2u;

@compute @workgroup_size(64)
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
    lane = subgroup_invocation_id;

    let index = subgroup_invocation_id + subgroup_size * subgroup_id;
    let linear_block_32x32_index = index >> 4u;
    var block32x32_index = 2 * vec2<i32>(wg_id);
    block32x32_index.x += i32(extractBits(index, 4u, 1u));
    block32x32_index.y += i32(extractBits(index, 5u, 1u));
    let local_block_index = vec2<i32>(i32(extractBits(index, 0u, 2u)), i32(extractBits(index, 2u, 2u)));
    let block8x8_index = 4 * block32x32_index + local_block_index;

    var meta_entry = vec2<u32>(0u, 0u);
    var quant = 0;

    let in_range_8x8 = all(block8x8_index < regs.resolution_8x8_blocks);
    let in_range_32x32 = all(block32x32_index < regs.resolution_32x32_blocks);
    var num_bits_for_q = 0u;

    if (in_range_32x32)
    {
        let block_index = regs.block_offset_32x32 +
            regs.block_stride_32x32 * block32x32_index.y +
            block32x32_index.x;
        quant = quant_data[block_index];
    }

    if (in_range_8x8)
    {
        let block_index = u32(regs.block_offset_8x8 +
            regs.block_stride_8x8 * block8x8_index.y +
            block8x8_index.x);
        meta_entry = block_meta[block_index];
        let num_planes = block_stats[block_index * BLOCK_STATS_WORDS];
        num_bits_for_q = block_stats[block_index * BLOCK_STATS_WORDS + 1u + min(num_planes, u32(quant))] >> 16u;
    }

    let code_word = quantize_code_word(meta_entry.x, quant);
    let active_code_word = (code_word & 0xffffu) != 0u;

    let code_word_ballot = subgroupBallot(active_code_word);
    let cluster_first_lane = lane & ~15u;
    let local_ballot = extractBits(code_word_ballot[cluster_first_lane >> 5u], cluster_first_lane & 31u, 16u);

    let required_plane_bytes = compute_required_8x8_size(code_word);
    let required_sign_bits = num_bits_for_q - required_plane_bytes * 8u;

    var required_bits_with_meta = num_bits_for_q;
    if (required_bits_with_meta != 0u)
    {
        required_bits_with_meta += 24u;
    }

    let writes_header =
        all(block32x32_index < regs.resolution_32x32_blocks) && (index & 15u) == 15u;

    let payload_total_bits = clustered_add16(required_bits_with_meta);
    var payload_total_words = (payload_total_bits + 31u) / 32u;
    if (payload_total_words != 0u)
    {
        payload_total_words += HeaderSize;
    }

    var global_payload_offset = 0u;
    if (writes_header && payload_total_words != 0u)
    {
        global_payload_offset = atomicAdd(&payload_data[BITSTREAM_PAYLOAD_COUNTER], payload_total_words);
    }
    global_payload_offset = subgroupShuffle(global_payload_offset, lane | 15u);

    let total_subblocks = countOneBits(local_ballot);

    let total_sign_bits = inclusive_add_clustered16(required_sign_bits);
    let local_planes_offset = inclusive_add_clustered16(required_plane_bytes) - required_plane_bytes;
    var local_sign_offset = total_sign_bits - required_sign_bits;
    var global_planes_offset = 4u * global_payload_offset + 3u * total_subblocks + 4u * HeaderSize;
    let global_sign_offset = global_planes_offset + clustered_add16(required_plane_bytes);
    global_planes_offset += local_planes_offset;

    let total_sign_bytes = (subgroupShuffle(total_sign_bits, lane | 15u) + 7u) / 8u;

    // Everything past the header is ORed in, so clear this packet's words first.
    for (var w = HeaderSize + (index & 15u); w < payload_total_words; w += 16u)
    {
        atomicStore(&bitstream_data[global_payload_offset + w], 0u);
    }

    storageBarrier();

    if (writes_header)
    {
        let block_index = u32(regs.block_offset_32x32 +
            block32x32_index.y * regs.block_stride_32x32 + block32x32_index.x);

        if (payload_total_words != 0u)
        {
            atomicStore(&bitstream_data[global_payload_offset + 0u],
                local_ballot | (payload_total_words << 16u) | (regs.sequence_code << 28u));
            atomicStore(&bitstream_data[global_payload_offset + 1u],
                modify_quant_code(regs.quant_resolution_code, quant) | (block_index << 8u));
        }

        bitstream_meta[block_index] = vec2<u32>(global_payload_offset, payload_total_words);
    }

    // Followed by N code words which map to the local ballot of active 16x16 regions.
    if (active_code_word)
    {
        let block_header_offset = countOneBits(extractBits(
            local_ballot, 0u, u32(local_block_index.y * 4 + local_block_index.x)));

        let in_q_bits = extractBits(meta_entry.x, Q_PLANES_OFFSET, Q_PLANES_BITS);
        let out_q_bits = extractBits(code_word, Q_PLANES_OFFSET, Q_PLANES_BITS);
        var input_offset = meta_entry.y;
        var output_offset = global_planes_offset;

        for (var bit_offset = 0u; bit_offset < 16u; bit_offset += 2u)
        {
            let out_planes = extractBits(code_word, bit_offset, 2u) + out_q_bits;
            var in_planes = extractBits(meta_entry.x, bit_offset, 2u) + in_q_bits;
            if (in_planes != 0u)
            {
                in_planes++;
            }

            let sign_plane = read_payload_byte(input_offset);

            if (out_planes != 0u)
            {
                let significant_mask = copy_bytes(&output_offset, input_offset + 1u, out_planes);
                append_sign_plane(linear_block_32x32_index, &local_sign_offset, sign_plane, significant_mask);
            }

            input_offset += in_planes;
        }

        flush_bitstream_bytes();
        flush_sign_plane(linear_block_32x32_index, local_sign_offset);

        or_bitstream_bytes(2u * (2u * global_payload_offset + block_header_offset + 2u * HeaderSize),
                           code_word & 0xffffu, 2u);
        or_bitstream_bytes(4u * global_payload_offset + 2u * total_subblocks + block_header_offset + 4u * HeaderSize,
                           code_word >> 16u, 1u);
    }

    workgroupBarrier();

    // Copy out all sign planes for any given group.
    for (var i = index & 15u; i < total_sign_bytes / 4u; i += 16u)
    {
        let sign_word = atomicLoad(&shared_sign_bank[linear_block_32x32_index][i]);
        or_bitstream_bytes(global_sign_offset + 4u * i, sign_word, 4u);
    }

    // Copy out any stragglers.
    for (var i = (total_sign_bytes & ~3u) + (index & 15u); i < total_sign_bytes; i += 16u)
    {
        let sign_word = atomicLoad(&shared_sign_bank[linear_block_32x32_index][i / 4u]);
        or_bitstream_bytes(global_sign_offset + i, sign_word >> (8u * (i & 3u)), 1u);
    }
}
