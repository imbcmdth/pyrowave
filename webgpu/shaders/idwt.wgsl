// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/idwt.comp. Inverse CDF 9/7 transform producing one 32x32 tile.
// Needs common.wgsl and dwt_common.wgsl.
//
// The GLSL writes every level through an image2D. Here there are two entry points:
// - main writes the LL band of the next level up into the r32float pyramid.
// - main_final applies the DC shift and writes 8-bit output. r8unorm is not a storage
//   format in core WebGPU, so the planes are a plain buffer of packed bytes instead,
//   which is also what the CPU readback wants. Each thread packs four horizontally
//   adjacent pixels into one u32.
// - The components of a level run as one dispatch, one per workgroup_id.z, each with
//   its own registers, since WebGPU puts barriers between dispatches. The input is
//   the level's 12 layers, read from input_layer, and main writes the LL layer of
//   the next level up at output_layer.

struct Registers
{
    resolution: vec2<i32>,
    inv_resolution: vec2<f32>,
    store_fp16: u32,
    // main_final only, in u32 words and rows. The last tile can hang over the edge of
    // the plane, which image stores in the GLSL drop silently.
    output_offset: u32,
    output_stride: u32,
    output_rows: u32,
    input_layer: i32,
    output_layer: i32,
    padding0: u32,
    padding1: u32,
};

@group(0) @binding(0) var<storage, read> dispatch_registers: array<Registers>;
@group(0) @binding(1) var uTexture: texture_2d_array<f32>;
@group(0) @binding(2) var uSampler: sampler; // Nearest, mirror repeat.
@group(0) @binding(3) var uOutput: texture_storage_2d_array<r32float, write>;
@group(0) @binding(4) var<storage, read_write> uOutputPlane: array<u32>;

var<private> registers: Registers;

var<private> workgroup_id: vec2<i32>;

fn generate_mirror_uv(coord_in: vec2<i32>, even_x: bool, even_y: bool) -> vec2<f32>
{
    var coord = coord_in;
    coord -= vec2<i32>(vec2<bool>(even_x, even_y) & (coord < vec2<i32>(0)));
    coord += 1;
    coord += vec2<i32>(vec2<bool>(!even_x, !even_y) & (coord >= registers.resolution));
    let uv = vec2<f32>(coord) * registers.inv_resolution;
    return uv.yx; // Transpose on load.
}

fn write_shared_4x4(coord: vec2<i32>, texels0: vec4<f32>, texels1: vec4<f32>, texels2: vec4<f32>, texels3: vec4<f32>)
{
    store_shared(coord.y + 0, 2 * coord.x + 0, vec2<f32>(texels0.x, texels2.x));
    store_shared(coord.y + 0, 2 * coord.x + 1, vec2<f32>(texels1.x, texels3.x));
    store_shared(coord.y + 0, 2 * coord.x + 2, vec2<f32>(texels0.y, texels2.y));
    store_shared(coord.y + 0, 2 * coord.x + 3, vec2<f32>(texels1.y, texels3.y));
    store_shared(coord.y + 1, 2 * coord.x + 0, vec2<f32>(texels0.z, texels2.z));
    store_shared(coord.y + 1, 2 * coord.x + 1, vec2<f32>(texels1.z, texels3.z));
    store_shared(coord.y + 1, 2 * coord.x + 2, vec2<f32>(texels0.w, texels2.w));
    store_shared(coord.y + 1, 2 * coord.x + 3, vec2<f32>(texels1.w, texels3.w));
}

fn load_image_with_apron()
{
    let base_coord = workgroup_id * vec2<i32>(BLOCK_SIZE_HALF) - APRON_HALF;
    let local_coord0 = 2 * unswizzle8x8(local_index);
    let coord0 = base_coord + local_coord0;

    // Transpose on load.
    {
        let texels0 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(coord0, true, true), registers.input_layer + 0).wxzy);
        let texels1 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(coord0, false, true), registers.input_layer + 2).wxzy);
        let texels2 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(coord0, true, false), registers.input_layer + 1).wxzy);
        let texels3 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(coord0, false, false), registers.input_layer + 3).wxzy);
        write_shared_4x4(local_coord0, texels0, texels1, texels2, texels3);
    }

    let local_coord_horiz = vec2<i32>(BLOCK_SIZE_HALF + 2 * i32(local_index % 2u), 2 * i32(local_index / 2u));
    if (local_coord_horiz.y < BLOCK_SIZE_HALF + 2 * APRON_HALF)
    {
        let c = base_coord + local_coord_horiz;
        let texels0 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, true, true), registers.input_layer + 0).wxzy);
        let texels1 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, false, true), registers.input_layer + 2).wxzy);
        let texels2 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, true, false), registers.input_layer + 1).wxzy);
        let texels3 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, false, false), registers.input_layer + 3).wxzy);
        write_shared_4x4(local_coord_horiz, texels0, texels1, texels2, texels3);
    }

    let local_coord_vert = local_coord_horiz.yx;
    if (local_coord_vert.x < BLOCK_SIZE_HALF)
    {
        let c = base_coord + local_coord_vert;
        let texels0 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, true, true), registers.input_layer + 0).wxzy);
        let texels1 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, false, true), registers.input_layer + 2).wxzy);
        let texels2 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, true, false), registers.input_layer + 1).wxzy);
        let texels3 = mediump_texels(textureGather(0, uTexture, uSampler, generate_mirror_uv(c, false, false), registers.input_layer + 3).wxzy);
        write_shared_4x4(local_coord_vert, texels0, texels1, texels2, texels3);
    }

    workgroupBarrier();
}

const SIZE8: i32 = 8;
const PADDED_SIZE8: i32 = SIZE8 + 2 * APRON;
const PADDED_SIZE8_HALF: i32 = PADDED_SIZE8 / 2;

fn inverse_transform8x2()
{
    var values: array<vec2<f32>, PADDED_SIZE8>;

    let local_coord = vec2<i32>(8 * i32(local_index % 4u), i32(local_index / 4u));

    for (var i = 0; i < PADDED_SIZE8; i += 2)
    {
        let v0 = load_shared(local_coord.y, local_coord.x + i + 0);
        let v1 = load_shared(local_coord.y, local_coord.x + i + 1);
        values[i + 0] = v0 * K;
        values[i + 1] = v1 * inv_K;
    }

    // CDF 9/7 lifting steps.
    for (var i = 2; i < PADDED_SIZE8 - 1; i += 2) { values[i] -= DELTA * (values[i - 1] + values[i + 1]); }
    for (var i = 3; i < PADDED_SIZE8 - 2; i += 2) { values[i] -= GAMMA * (values[i - 1] + values[i + 1]); }
    for (var i = 4; i < PADDED_SIZE8 - 3; i += 2) { values[i] -= BETA * (values[i - 1] + values[i + 1]); }
    for (var i = 5; i < PADDED_SIZE8 - 4; i += 2) { values[i] -= ALPHA * (values[i - 1] + values[i + 1]); }

    // Avoid WAR hazard.
    workgroupBarrier();

    for (var i = APRON_HALF; i < PADDED_SIZE8_HALF - APRON_HALF; i++)
    {
        let a = values[2 * i + 0];
        let b = values[2 * i + 1];

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

fn inverse_transform4x2(active_lane: bool, y_offset: i32)
{
    var values: array<vec2<f32>, PADDED_SIZE4>;

    let local_coord = vec2<i32>(4 * i32(local_index % 8u), i32(local_index / 8u) + y_offset);

    if (active_lane)
    {
        for (var i = 0; i < PADDED_SIZE4; i += 2)
        {
            let v0 = load_shared(local_coord.y, local_coord.x + i + 0);
            let v1 = load_shared(local_coord.y, local_coord.x + i + 1);
            values[i + 0] = v0 * K;
            values[i + 1] = v1 * inv_K;
        }

        // CDF 9/7 lifting steps.
        for (var i = 2; i < PADDED_SIZE4 - 1; i += 2) { values[i] -= DELTA * (values[i - 1] + values[i + 1]); }
        for (var i = 3; i < PADDED_SIZE4 - 2; i += 2) { values[i] -= GAMMA * (values[i - 1] + values[i + 1]); }
        for (var i = 4; i < PADDED_SIZE4 - 3; i += 2) { values[i] -= BETA * (values[i - 1] + values[i + 1]); }
        for (var i = 5; i < PADDED_SIZE4 - 4; i += 2) { values[i] -= ALPHA * (values[i - 1] + values[i + 1]); }
    }

    // Avoid WAR hazard.
    workgroupBarrier();

    if (active_lane)
    {
        for (var i = APRON_HALF; i < PADDED_SIZE4_HALF - APRON_HALF; i++)
        {
            let a = values[2 * i + 0];
            let b = values[2 * i + 1];

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

fn transform_tile(local_invocation_index: u32, wg_id: vec3<u32>)
{
    local_index = local_invocation_index;
    workgroup_id = vec2<i32>(wg_id.xy);
    registers = dispatch_registers[wg_id.z];

    load_image_with_apron();

    // Horizontal transform.
    inverse_transform8x2();

    // Also need to transform the apron.
    inverse_transform4x2(local_index < 32u, BLOCK_SIZE_HALF);

    workgroupBarrier();

    // Vertical transform.
    inverse_transform8x2();

    workgroupBarrier();
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(workgroup_id) wg_id: vec3<u32>)
{
    transform_tile(local_invocation_index, wg_id);

    let fp16 = registers.store_fp16 != 0u;
    let local_coord = unswizzle8x8(local_index);

    for (var y = local_coord.y; y < BLOCK_SIZE_HALF; y += 8)
    {
        for (var x = local_coord.x; x < BLOCK_SIZE; x += 8)
        {
            let v = load_shared(y, x);
            textureStore(uOutput, vec2<i32>(2 * y + 0, x) + BLOCK_SIZE * workgroup_id.yx, registers.output_layer,
                         vec4<f32>(round_wavelet(v.x, fp16)));
            textureStore(uOutput, vec2<i32>(2 * y + 1, x) + BLOCK_SIZE * workgroup_id.yx, registers.output_layer,
                         vec4<f32>(round_wavelet(v.y, fp16)));
        }
    }
}

@compute @workgroup_size(64)
fn main_final(@builtin(local_invocation_index) local_invocation_index: u32,
              @builtin(workgroup_id) wg_id: vec3<u32>)
{
    transform_tile(local_invocation_index, wg_id);

    // Shared row y holds output pixels (2y, 2y + 1) of output row x, so four horizontally
    // adjacent pixels 4m .. 4m + 3 come from shared rows 2m and 2m + 1.
    for (var i = local_index; i < u32(BLOCK_SIZE * BLOCK_SIZE / 4); i += 64u)
    {
        let m = i32(i & 7u);
        let x = i32(i >> 3u);
        let a = load_shared(2 * m + 0, x) + 0.5;
        let b = load_shared(2 * m + 1, x) + 0.5;

        let row = u32(BLOCK_SIZE * workgroup_id.x + x);
        let word = u32(BLOCK_SIZE / 4 * workgroup_id.y + m);
        if (row < registers.output_rows && word < registers.output_stride)
        {
            uOutputPlane[registers.output_offset + row * registers.output_stride + word] =
                pack4x8unorm(vec4<f32>(a.x, a.y, b.x, b.y));
        }
    }
}
