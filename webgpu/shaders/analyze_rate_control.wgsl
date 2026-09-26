// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/analyze_rate_control.comp. Needs subgroup.wgsl.
//
// Differences from the GLSL:
// - The bucket buffer is one atomic u32 array addressed by word, since WGSL cannot
//   mix atomic and plain members in a runtime sized buffer. RDOperation entries are
//   written with atomicStore.
// - QuantStats {float16_t, uint16_t} is one u32: FP16 bits low, cost high.
// - All bands are one dispatch, see band_dispatch.wgsl.
// Like the GLSL this needs subgroups of 16 to 64 lanes.

struct Registers
{
    resolution: vec2<i32>,
    resolution_8x8_blocks: vec2<i32>,
    block_offset_8x8: i32,
    block_stride_8x8: i32,
    block_offset_32x32: i32,
    block_stride_32x32: i32,
    total_wg_count: u32,
    num_blocks_aligned: u32,
    block_index_shamt: u32,
    padding: u32,
};

// struct BlockStats { uint num_planes; QuantStats errors[15]; }, as a flat word array.
// errors[15] is reachable below and reads the next block's first word, as in the GLSL.
const BLOCK_STATS_WORDS: u32 = 16u;

const BLOCK_SPACE_SUBDIVISION: u32 = 16u;

// Word offsets into the Buckets buffer:
//   uint count;                      // 0
//   uint consumed_payload;           // 1
//   layout(offset = 64) uint total_savings_per_bucket[128 * BLOCK_SPACE_SUBDIVISION];
//   RDOperation rdo_operations[];   // { int quant; uint block_offset_saving; }
const BUCKETS_CONSUMED_PAYLOAD: u32 = 1u;
const BUCKETS_TOTAL_SAVINGS: u32 = 16u;
const BUCKETS_RDO_OPERATIONS: u32 = BUCKETS_TOTAL_SAVINGS + 128u * BLOCK_SPACE_SUBDIVISION;

@group(0) @binding(0) var<storage, read> band_registers: array<Registers>;
@group(0) @binding(1) var<storage, read_write> buckets: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> block_stats: array<u32>;

var<workgroup> shared_rate_cost: array<u32, 16>;
var<workgroup> shared_distortion: array<f32, 16>;

var<private> registers: Registers;
var<private> lane: u32;
var<private> workgroup_id: vec2<i32>;

// Perform operations that cause lower distortion first.
fn distortion_to_bucket_index(d: f32, cost: f32, d_base: f32, cost_base: f32) -> u32
{
    if (cost == cost_base)
    {
        return 0u;
    }

    // Compress a large range into 64 possible buckets.
    // Every band is ~1.5 dB.
    // Greedily chase least added (weighted) distortion per byte removed from code stream.
    let index = 60.0 + 2.0 * log2(max(d - d_base, 0.0) / (cost_base - cost));
    return u32(max(index + 0.5, 0.0));
}

fn inclusive_max_clustered16(v_in: u32) -> u32
{
    // Ensures that we never end up with a value > 127.
    var v = min(v_in, 128u - 16u + lane);

    for (var i = 1u; i < 16u; i *= 2u)
    {
        // Ensure monotonic progression for buckets.
        // Separate every quant level out by at least one bucket.
        let up = subgroupShuffleUp(v, i) + i;
        v = max(v, select(0u, up, lane >= i));
    }

    return v;
}

fn emit_rdo_operations()
{
    var distortion: f32;
    var cost: f32;

    if (lane < 16u)
    {
        cost = f32(shared_rate_cost[lane]);
        distortion = shared_distortion[lane];
    }
    else
    {
        // Dummy values.
        cost = f32(shared_rate_cost[15]);
        distortion = 1e30;
    }

    var bucket_index = distortion_to_bucket_index(distortion, cost, shared_distortion[0], f32(shared_rate_cost[0]));
    if (lane == 0u)
    {
        bucket_index = 0u;
    }

    // Constraints:
    // bucket_index for Q1 must be less than bucket_index for Q2 if Q1 < Q2.
    // If a high quant target sees very favorable RD, lower bucket indices for lower Q values.
    let inclusive_bucket_index = inclusive_max_clustered16(bucket_index);

    if (lane == 0u)
    {
        let unquantized_cost = shared_rate_cost[0];
        atomicAdd(&buckets[BUCKETS_CONSUMED_PAYLOAD], unquantized_cost);
    }
    else if (lane < 16u)
    {
        let saving = shared_rate_cost[lane - 1u] - shared_rate_cost[lane];

        if (saving != 0u)
        {
            let block32x32_index = workgroup_id;
            let block_index = registers.block_offset_32x32 +
                block32x32_index.y * registers.block_stride_32x32 + block32x32_index.x;
            let subdivision = u32(block_index) >> registers.block_index_shamt;
            atomicAdd(&buckets[BUCKETS_TOTAL_SAVINGS + inclusive_bucket_index * BLOCK_SPACE_SUBDIVISION + subdivision], saving);
            let op_index = u32(block_index) + inclusive_bucket_index * registers.num_blocks_aligned;
            atomicStore(&buckets[BUCKETS_RDO_OPERATIONS + 2u * op_index + 0u], lane);
            atomicStore(&buckets[BUCKETS_RDO_OPERATIONS + 2u * op_index + 1u], u32(block_index) | (saving << 16u));
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) num_workgroups: vec3<u32>,
        @builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
        @builtin(subgroup_size) subgroup_size: u32)
{
    let band = select_band(wg, num_workgroups);
    if (!band.valid)
    {
        return;
    }
    registers = band_registers[band.band];

    let subgroup_id = allocate_subgroup_id(local_invocation_index, subgroup_invocation_id);
    lane = subgroup_invocation_id;
    workgroup_id = vec2<i32>(band.id);

    // Each workgroup processes a 64x64 block and computes all possible rate wins for every potential quant rate.
    let index = subgroup_invocation_id + subgroup_size * subgroup_id;
    let block32x32_index = workgroup_id;
    let local_block_index = vec2<i32>(i32(extractBits(index, 0u, 2u)), i32(extractBits(index, 2u, 2u)));
    let block8x8_index = 4 * block32x32_index + local_block_index;

    var num_active_planes = 0u;

    let block8x8_in_range = all(block8x8_index < registers.resolution_8x8_blocks);
    let block_index_8x8 = registers.block_offset_8x8 +
        registers.block_stride_8x8 * block8x8_index.y +
        block8x8_index.x;

    if (block8x8_in_range)
    {
        num_active_planes = block_stats[u32(block_index_8x8) * BLOCK_STATS_WORDS];
    }

    let bit_index = index >> 4u;

    for (var i = bit_index; i < 16u; i += 4u)
    {
        var dist = 0.0;
        var cost = 0u;

        if (block8x8_in_range)
        {
            let stats = block_stats[u32(block_index_8x8) * BLOCK_STATS_WORDS + 1u + min(i, num_active_planes)];
            dist = unpack2x16float(stats).x;
            cost = stats >> 16u;
        }

        // 16 bits to encode the control codes, 8 bits to encode Q bits + quant scale.
        // Cost is encoded in terms of bits. 8x8 blocks are decoded in isolation.
        if (cost != 0u)
        {
            cost += 24u;
        }

        if (subgroup_size == 16u)
        {
            cost = subgroupAdd(cost);
            dist = subgroupAdd(dist);
        }
        else
        {
            cost += subgroupShuffleXor(cost, 1u);
            cost += subgroupShuffleXor(cost, 2u);
            cost += subgroupShuffleXor(cost, 4u);
            cost += subgroupShuffleXor(cost, 8u);

            dist += subgroupShuffleXor(dist, 1u);
            dist += subgroupShuffleXor(dist, 2u);
            dist += subgroupShuffleXor(dist, 4u);
            dist += subgroupShuffleXor(dist, 8u);
        }

        if ((index & 15u) == 0u)
        {
            // Need to encode a header.
            // We can eliminate 32x32 blocks if everything decodes to 0.
            if (cost != 0u)
            {
                cost += 64u;
            }

            // Each packet is aligned to 4 bytes for practical reasons.
            shared_rate_cost[i] = (cost + 31u) >> 5u;
            shared_distortion[i] = dist;
        }
    }

    workgroupBarrier();

    if (subgroup_id == 0u)
    {
        emit_rdo_operations();
    }
}
