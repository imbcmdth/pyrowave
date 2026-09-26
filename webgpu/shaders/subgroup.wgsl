// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// The GLSL shaders index their work by
//     gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID
// rather than gl_LocalInvocationIndex, so that lanes which cooperate through subgroup
// operations (clusters of 8 or 16 consecutive indices) are guaranteed to share a
// subgroup. Core WGSL has subgroup_invocation_id and subgroup_size but no subgroup_id
// (that is the optional subgroup_id language extension), and WebGPU makes no promise
// about how local_invocation_index maps onto subgroups. Instead, every subgroup takes
// a ticket from a workgroup counter. The ticket order is arbitrary, which is fine: all
// the shaders need is a bijection that keeps each subgroup's indices contiguous.
//
// This assumes full subgroups, which holds for the power of two workgroup sizes used
// here whenever the subgroup is no wider than the workgroup.

var<workgroup> subgroup_ticket: atomic<u32>;

// Must be called from uniform control flow, before anything else touches
// subgroup_ticket. WGSL promises zero initialized workgroup memory, but wgpu-native
// v29 on Vulkan was seen to leave the counter running across workgroups, so it is
// reset explicitly.
fn allocate_subgroup_id(local_invocation_index: u32, lane: u32) -> u32
{
    if (local_invocation_index == 0u)
    {
        atomicStore(&subgroup_ticket, 0u);
    }
    workgroupBarrier();

    var id = 0u;
    if (lane == 0u)
    {
        id = atomicAdd(&subgroup_ticket, 1u);
    }
    return subgroupBroadcast(id, 0u);
}
