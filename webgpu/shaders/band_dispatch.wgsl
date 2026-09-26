// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// The Vulkan encoder records one dispatch per (level, component, band), about 40 per
// stage, with no barriers between them since they touch disjoint data. WebGPU
// implementations put a barrier between every pair of dispatches that write storage,
// and on small bands that overhead dominates. So the host batches every band of a
// stage into a single dispatch instead: the workgroups are numbered linearly, band
// after band, and each workgroup looks up which band it belongs to and that band's
// push constant block.
//
// band_dispatch[i] = (first linear workgroup, workgroups in x, workgroup count, 0)
// for band i, sorted by first workgroup. The shader declares its own
// band_registers: array<Registers> at binding 0 with the matching entries.

@group(0) @binding(7) var<storage, read> band_dispatch: array<vec4<u32>>;

struct BandWorkgroup
{
    band: u32,
    id: vec2<u32>,
    valid: bool,
};

fn select_band(wg_id: vec3<u32>, num_workgroups: vec3<u32>) -> BandWorkgroup
{
    let linear = wg_id.x + wg_id.y * num_workgroups.x;
    var band = 0u;
    let count = arrayLength(&band_dispatch);
    for (var i = 1u; i < count; i++)
    {
        if (linear >= band_dispatch[i].x)
        {
            band = i;
        }
    }

    let info = band_dispatch[band];
    let local_id = linear - info.x;

    var result: BandWorkgroup;
    result.band = band;
    result.id = vec2<u32>(local_id % info.y, local_id / info.y);
    result.valid = local_id < info.z;
    return result;
}
