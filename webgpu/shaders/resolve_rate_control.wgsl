// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/resolve_rate_control.comp. Needs subgroup.wgsl.
//
// The GLSL makes the workgroup exactly one subgroup (its size is a specialization
// constant that the host sets to the subgroup size it forces), so one
// subgroupInclusiveAdd scans a whole chunk of operations. WebGPU can neither force a
// subgroup size nor size a workgroup from one, so this uses a fixed workgroup of 64
// and extends the scan across subgroups through workgroup memory.
//
// The chunk size does not change the result. An operation is applied exactly when
// the savings of all operations before it in the bucket fall short of the target;
// the chunking only decides how early the loop can stop.

struct Registers
{
    target_payload_size: u32,
    num_blocks_per_subdivision: u32,
    padding0: u32,
    padding1: u32,
};

const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_SPACE_SUBDIVISION: u32 = 16u;
const BUCKETS_CONSUMED_PAYLOAD: u32 = 1u;
const BUCKETS_TOTAL_SAVINGS: u32 = 16u;
const BUCKETS_RDO_OPERATIONS: u32 = BUCKETS_TOTAL_SAVINGS + 128u * BLOCK_SPACE_SUBDIVISION;

@group(0) @binding(0) var<uniform> registers: Registers;
@group(0) @binding(1) var<storage, read> buckets: array<u32>;
@group(0) @binding(2) var<storage, read_write> quant_data: array<atomic<i32>>;

var<workgroup> shared_subgroup_totals: array<u32, WORKGROUP_SIZE / 4u>;
var<workgroup> shared_chunk_total: u32;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
        @builtin(subgroup_size) subgroup_size: u32)
{
    let subgroup_id = allocate_subgroup_id(local_invocation_index, subgroup_invocation_id);
    let local_index = subgroup_id * subgroup_size + subgroup_invocation_id;
    let bucket = wg_id.x;

    var required_savings_per_bucket = i32(buckets[BUCKETS_CONSUMED_PAYLOAD]) - i32(registers.target_payload_size);
    if (bucket != 0u)
    {
        let prev_bucket_total = i32(buckets[BUCKETS_TOTAL_SAVINGS + bucket - 1u]);
        // This bucket is empty.
        if (i32(buckets[BUCKETS_TOTAL_SAVINGS + bucket]) == prev_bucket_total)
        {
            return;
        }

        required_savings_per_bucket -= prev_bucket_total;
    }
    else
    {
        // This bucket is empty.
        if (buckets[BUCKETS_TOTAL_SAVINGS + bucket] == 0u)
        {
            return;
        }
    }

    // If all previous buckets can complete the job, skip.
    if (required_savings_per_bucket <= 0)
    {
        return;
    }

    let required = u32(required_savings_per_bucket);
    var total_saved = 0u;

    for (var i = 0u; i < registers.num_blocks_per_subdivision && total_saved < required; i += WORKGROUP_SIZE)
    {
        var op = vec2<u32>(0u, 0u);
        if (i + local_index < registers.num_blocks_per_subdivision)
        {
            let op_index = bucket * registers.num_blocks_per_subdivision + i + local_index;
            op = vec2<u32>(buckets[BUCKETS_RDO_OPERATIONS + 2u * op_index + 0u],
                           buckets[BUCKETS_RDO_OPERATIONS + 2u * op_index + 1u]);
        }

        let saving = extractBits(op.y, 16u, 16u);
        let block_offset = extractBits(op.y, 0u, 16u);

        var scan_saving = subgroupInclusiveAdd(saving);
        if (subgroup_invocation_id == subgroup_size - 1u)
        {
            shared_subgroup_totals[subgroup_id] = scan_saving;
        }

        workgroupBarrier();

        for (var s = 0u; s < subgroup_id; s++)
        {
            scan_saving += shared_subgroup_totals[s];
        }

        if (local_index == WORKGROUP_SIZE - 1u)
        {
            shared_chunk_total = scan_saving;
        }

        let should_apply_quant = total_saved + scan_saving - saving < required;
        if (should_apply_quant && saving != 0u)
        {
            atomicMax(&quant_data[block_offset], i32(op.x));
        }

        // Also orders the shared_subgroup_totals reads above before the next chunk's writes.
        total_saved += workgroupUniformLoad(&shared_chunk_total);
    }
}
