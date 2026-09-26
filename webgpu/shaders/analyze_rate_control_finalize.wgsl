// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/analyze_rate_control_finalize.comp.
//
// The GLSL runs 512 invocations. The default maxComputeInvocationsPerWorkgroup in
// WebGPU is 256, so each invocation here plays two of the GLSL's, u and u + 256, with
// the same scan steps and barriers.
//
// Note the scan loop stops at step < 256 (gl_WorkGroupSize.x / 2 in the GLSL), so
// each entry sums a window of the last 256 invocations rather than being a full
// prefix sum over all 512. That is kept as is so both backends make the same rate
// control decisions.

const BLOCK_SPACE_SUBDIVISION: u32 = 16u;
const NUM_ENTRIES: u32 = 128u * BLOCK_SPACE_SUBDIVISION / 4u; // uvec4 entries, 512.
const BUCKETS_TOTAL_SAVINGS: u32 = 16u; // layout(offset = 64)

@group(0) @binding(0) var<storage, read_write> buckets: array<u32>;

var<workgroup> shared_scan: array<u32, NUM_ENTRIES>;

fn load_entry(u: u32) -> vec4<u32>
{
    let base = BUCKETS_TOTAL_SAVINGS + 4u * u;
    return vec4<u32>(buckets[base + 0u], buckets[base + 1u], buckets[base + 2u], buckets[base + 3u]);
}

fn store_entry(u: u32, v: vec4<u32>)
{
    let base = BUCKETS_TOTAL_SAVINGS + 4u * u;
    buckets[base + 0u] = v.x;
    buckets[base + 1u] = v.y;
    buckets[base + 2u] = v.z;
    buckets[base + 3u] = v.w;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local_index: u32)
{
    var v: array<vec4<u32>, 2>;

    for (var k = 0u; k < 2u; k++)
    {
        let u = local_index + 256u * k;
        var e = load_entry(u);
        e.y += e.x;
        e.z += e.y;
        e.w += e.z;
        v[k] = e;
        shared_scan[u] = e.w;
    }

    workgroupBarrier();

    for (var step = 1u; step < NUM_ENTRIES / 2u; step *= 2u)
    {
        workgroupBarrier();

        var shuffled_up = array<u32, 2>(0u, 0u);
        for (var k = 0u; k < 2u; k++)
        {
            let u = local_index + 256u * k;
            if (u >= step)
            {
                shuffled_up[k] = shared_scan[u - step];
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < 2u; k++)
        {
            let u = local_index + 256u * k;
            v[k] += vec4<u32>(shuffled_up[k]);
            shared_scan[u] = v[k].w;
        }
    }

    for (var k = 0u; k < 2u; k++)
    {
        store_entry(local_index + 256u * k, v[k]);
    }
}
