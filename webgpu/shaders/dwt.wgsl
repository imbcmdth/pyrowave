// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/dwt.comp. Forward CDF 9/7 transform of one 32x32 tile.
// Needs common.wgsl and dwt_common.wgsl.
//
// Differences from the GLSL:
// - The GLSL derives its thread index from the subgroup only to line the swizzle up
//   with subgroups for speed; nothing depends on it, so this uses
//   local_invocation_index and needs no subgroup support.
// - Output is r32float, rounded through FP16 where the GLSL stores R16F.

struct Registers
{
    resolution: vec2<i32>,
    inv_resolution: vec2<f32>,
    aligned_resolution: vec2<i32>,
    store_fp16: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> registers: Registers;
@group(0) @binding(1) var uTexture: texture_2d<f32>;
@group(0) @binding(2) var uSampler: sampler; // Nearest, mirror repeat.
@group(0) @binding(3) var uOutput: texture_storage_2d_array<r32float, write>;

override DCShift: bool = false;

var<private> workgroup_id: vec2<i32>;

fn generate_mirror_uv(coord_in: vec2<i32>) -> vec2<f32>
{
    var coord = coord_in;
    coord -= vec2<i32>(coord < vec2<i32>(0));
    coord += 1;
    let end_mirrored_clamp = (2 * registers.aligned_resolution) - registers.resolution;
    let past_wrapped_coord = coord + 2 * (registers.resolution - registers.aligned_resolution) + 1;
    coord = select(min(coord, registers.resolution), past_wrapped_coord, coord >= end_mirrored_clamp);

    return vec2<f32>(coord) * registers.inv_resolution;
}

fn gather_texels(coord: vec2<i32>) -> vec4<f32>
{
    var texels = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(coord)).wzxy);
    if (DCShift)
    {
        texels -= vec4<f32>(0.5);
    }
    return texels;
}

fn load_image_with_apron()
{
    let base_coord = workgroup_id * vec2<i32>(BLOCK_SIZE, BLOCK_SIZE) - APRON;
    let local_coord0 = 2 * unswizzle8x8(local_index);
    let coord0 = base_coord + local_coord0;

    let texels0 = gather_texels(coord0);
    let texels1 = gather_texels(coord0 + vec2<i32>(16, 0));
    let texels2 = gather_texels(coord0 + vec2<i32>(0, 16));
    let texels3 = gather_texels(coord0 + vec2<i32>(16, 16));

    let local_coord0_y_half = local_coord0.y >> 1u;

    // Pack two lines together in one vec2. This allows packed FP16 math easily by processing two lines in parallel.
    store_shared(local_coord0_y_half + 0, local_coord0.x + 0, texels0.xz);
    store_shared(local_coord0_y_half + 0, local_coord0.x + 1, texels0.yw);
    store_shared(local_coord0_y_half + 0, local_coord0.x + 16, texels1.xz);
    store_shared(local_coord0_y_half + 0, local_coord0.x + 17, texels1.yw);
    store_shared(local_coord0_y_half + 8, local_coord0.x + 0, texels2.xz);
    store_shared(local_coord0_y_half + 8, local_coord0.x + 1, texels2.yw);
    store_shared(local_coord0_y_half + 8, local_coord0.x + 16, texels3.xz);
    store_shared(local_coord0_y_half + 8, local_coord0.x + 17, texels3.yw);

    // Load the top-right apron
    {
        let local_coord = vec2<i32>(BLOCK_SIZE + 2 * i32(local_index % 4u), 2 * i32(local_index / 4u));
        let texels = gather_texels(base_coord + local_coord);
        store_shared(local_coord.y >> 1u, local_coord.x + 0, texels.xz);
        store_shared(local_coord.y >> 1u, local_coord.x + 1, texels.yw);
    }

    // Load the bottom-left apron
    {
        let local_coord = vec2<i32>(2 * i32(local_index % 16u), BLOCK_SIZE + 2 * i32(local_index / 16u));
        let texels = gather_texels(base_coord + local_coord);
        store_shared(local_coord.y >> 1u, local_coord.x + 0, texels.xz);
        store_shared(local_coord.y >> 1u, local_coord.x + 1, texels.yw);
    }

    if (local_index < 16u)
    {
        // Load the bottom-right apron
        let local_coord = vec2<i32>(BLOCK_SIZE + 2 * i32(local_index % 4u), BLOCK_SIZE + 2 * i32(local_index / 4u));
        let texels = gather_texels(base_coord + local_coord);
        store_shared(local_coord.y >> 1u, local_coord.x + 0, texels.xz);
        store_shared(local_coord.y >> 1u, local_coord.x + 1, texels.yw);
    }
}

const SIZE8: i32 = 8;
const PADDED_SIZE8: i32 = SIZE8 + 2 * APRON;
const PADDED_SIZE8_HALF: i32 = PADDED_SIZE8 / 2;

fn forward_transform8x2()
{
    var values: array<vec2<f32>, PADDED_SIZE8>;

    let local_coord = vec2<i32>(8 * i32(local_index % 4u), i32(local_index / 4u));

    for (var i = 0; i < PADDED_SIZE8; i++)
    {
        values[i] = load_shared(local_coord.y, local_coord.x + i);
    }

    // CDF 9/7 lifting steps.
    for (var i = 1; i < PADDED_SIZE8 - 1; i += 2) { values[i] += ALPHA * (values[i - 1] + values[i + 1]); }
    for (var i = 2; i < PADDED_SIZE8 - 2; i += 2) { values[i] += BETA * (values[i - 1] + values[i + 1]); }
    for (var i = 3; i < PADDED_SIZE8 - 3; i += 2) { values[i] += GAMMA * (values[i - 1] + values[i + 1]); }
    for (var i = 4; i < PADDED_SIZE8 - 4; i += 2) { values[i] += DELTA * (values[i - 1] + values[i + 1]); }

    // Avoid WAR hazard.
    workgroupBarrier();

    for (var i = APRON_HALF; i < PADDED_SIZE8_HALF - APRON_HALF; i++)
    {
        // Filter kernel rescale.
        let a = values[2 * i + 0] * inv_K;
        let b = values[2 * i + 1] * K;

        // Transpose the 2x2 block.
        let t0 = vec2<f32>(a.x, b.x);
        let t1 = vec2<f32>(a.y, b.y);

        // Transpose write
        let y_coord = (local_coord.x >> 1u) + (i - APRON_HALF);
        store_shared(y_coord, 2 * local_coord.y + 0, t0);
        store_shared(y_coord, 2 * local_coord.y + 1, t1);
    }
}

const SIZE4: i32 = 4;
const PADDED_SIZE4: i32 = SIZE4 + 2 * APRON;
const PADDED_SIZE4_HALF: i32 = PADDED_SIZE4 / 2;

fn forward_transform4x2(active_lane: bool, y_offset: i32)
{
    var values: array<vec2<f32>, PADDED_SIZE4>;

    let local_coord = vec2<i32>(4 * i32(local_index % 8u), i32(local_index / 8u) + y_offset);

    if (active_lane)
    {
        for (var i = 0; i < PADDED_SIZE4; i++)
        {
            values[i] = load_shared(local_coord.y, local_coord.x + i);
        }

        // CDF 9/7 lifting steps.
        for (var i = 1; i < PADDED_SIZE4 - 1; i += 2) { values[i] += ALPHA * (values[i - 1] + values[i + 1]); }
        for (var i = 2; i < PADDED_SIZE4 - 2; i += 2) { values[i] += BETA * (values[i - 1] + values[i + 1]); }
        for (var i = 3; i < PADDED_SIZE4 - 3; i += 2) { values[i] += GAMMA * (values[i - 1] + values[i + 1]); }
        for (var i = 4; i < PADDED_SIZE4 - 4; i += 2) { values[i] += DELTA * (values[i - 1] + values[i + 1]); }
    }

    // Avoid WAR hazard.
    workgroupBarrier();

    if (active_lane)
    {
        for (var i = APRON_HALF; i < PADDED_SIZE4_HALF - APRON_HALF; i++)
        {
            // Filter kernel rescale.
            let a = values[2 * i + 0] * inv_K;
            let b = values[2 * i + 1] * K;

            // Transpose the 2x2 block.
            let t0 = vec2<f32>(a.x, b.x);
            let t1 = vec2<f32>(a.y, b.y);

            // Transpose write
            let y_coord = (local_coord.x >> 1u) + (i - APRON_HALF);
            store_shared(y_coord, 2 * local_coord.y + 0, t0);
            store_shared(y_coord, 2 * local_coord.y + 1, t1);
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(workgroup_id) wg_id: vec3<u32>)
{
    local_index = local_invocation_index;
    workgroup_id = vec2<i32>(wg_id.xy);

    load_image_with_apron();

    workgroupBarrier();

    // Horizontal transform.
    forward_transform8x2();

    // Also need to transform the apron.
    forward_transform4x2(local_index < 32u, BLOCK_SIZE_HALF);

    workgroupBarrier();

    // Vertical transform.
    forward_transform8x2();

    workgroupBarrier();

    let fp16 = registers.store_fp16 != 0u;
    let local_coord = unswizzle8x8(local_index);
    for (var y = local_coord.y; y < BLOCK_SIZE_HALF; y += 8)
    {
        for (var x = local_coord.x * 2; x < BLOCK_SIZE; x += 16)
        {
            let v0 = load_shared(y, x + 0);
            let v1 = load_shared(y, x + 1);

            let img_x = x >> 1u;
            let img_y = y;

            let base_image_coord = workgroup_id * (BLOCK_SIZE / 2) + vec2<i32>(img_x, img_y);
            textureStore(uOutput, base_image_coord, 0, vec4<f32>(round_wavelet(v0.x, fp16)));
            textureStore(uOutput, base_image_coord, 2, vec4<f32>(round_wavelet(v0.y, fp16)));
            textureStore(uOutput, base_image_coord, 1, vec4<f32>(round_wavelet(v1.x, fp16)));
            textureStore(uOutput, base_image_coord, 3, vec4<f32>(round_wavelet(v1.y, fp16)));
        }
    }
}
