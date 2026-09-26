// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// WGSL port of shaders/dwt_common.h.
//
// The lifting math always runs in FP32 here (PRECISION 1 and 2 of the GLSL). What
// PRECISION changes is the type of the shared memory tile, and WGSL cannot pick a type
// with a constant, so the host prepends one of these to the shader source:
//   PRECISION 1: alias SHARED_VEC2 = u32;       packed with pack2x16float
//   PRECISION 2: alias SHARED_VEC2 = vec2<f32>;
// together with matching shared_pack() / shared_unpack() functions.
// PRECISION 1 is the same as the GLSL !FP16 path, which packs with packHalf2x16.

const APRON: i32 = 4;
const APRON_HALF: i32 = APRON / 2;
const BLOCK_SIZE: i32 = 32;
const BLOCK_SIZE_HALF: i32 = BLOCK_SIZE >> 1;

const ALPHA: f32 = -1.586134342059924;
const BETA: f32 = -0.052980118572961;
const GAMMA: f32 = 0.882911075530934;
const DELTA: f32 = 0.443506852043971;
const K: f32 = 1.230174104914001;
const inv_K: f32 = 1.0 / 1.230174104914001;

var<workgroup> shared_block: array<array<SHARED_VEC2, (BLOCK_SIZE + 2 * APRON) + 1>, (BLOCK_SIZE + 2 * APRON) / 2>;

fn load_shared(y: i32, x: i32) -> vec2<f32>
{
    return shared_unpack(shared_block[y][x]);
}

fn store_shared(y: i32, x: i32, v: vec2<f32>)
{
    shared_block[y][x] = shared_pack(v);
}

var<private> local_index: u32;
