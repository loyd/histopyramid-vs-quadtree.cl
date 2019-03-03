typedef float3 point_t;
typedef float4 cell_t;

inline uint c2id(uint2 coords, uint size) {
    return size * coords.y + coords.x;
}

inline void atomic_add_float(volatile global float *addr, float val) {
    union {
        uint u32;
        float f32;
    } next, expected, current;

    current.f32 = *addr;

    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile global uint *)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

/*
 * Pyramid memory layout:
 *
 * level |  0  |  1  |  2  |  3  | ... |  depth - 1  |
 * ------+-----+-----+-----+-----+-----+-------------+
 * cells | 1x1 | 2x2 | 4x4 | 8x8 | ... | size x size |
 *                                     '---- grid ---'
 */

kernel void make_grid(
    float4 bbox,
    uint size,
    global const point_t *restrict points,
    global float *restrict grid
) {
    int gid = get_global_id(0);

    point_t point = points[gid];

    float2 cell_sz = (bbox.zw - bbox.xy) / (float)size;

    float2 relative = clamp(point.xy, bbox.xy, bbox.zw) - bbox.xy;
    uint2 coords = convert_uint2(relative / cell_sz);

    uint idx = 4 * c2id(coords, size);

    atomic_add_float(grid + idx + 0, point.x);
    atomic_add_float(grid + idx + 1, point.y);
    atomic_add_float(grid + idx + 2, point.z);
    atomic_add_float(grid + idx + 3, 1.0f);
}

kernel void make_level(
    global const cell_t *restrict top,
    global cell_t *restrict current
) {
    uint cur_size = get_global_size(0);
    uint2 cur_coords = (uint2)(get_global_id(0), get_global_id(1));
    uint cur_id = c2id(cur_coords, cur_size);

    uint2 top_coords = 2 * cur_coords;
    uint top_size = 2 * cur_size;

    current[cur_id] = top[c2id(top_coords + (uint2)(0, 0), top_size)]
                    + top[c2id(top_coords + (uint2)(0, 1), top_size)]
                    + top[c2id(top_coords + (uint2)(1, 0), top_size)]
                    + top[c2id(top_coords + (uint2)(1, 1), top_size)];
}
