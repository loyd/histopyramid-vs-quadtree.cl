from __future__ import absolute_import

from os import path

import numpy as np
import pyopencl as cl
from pyopencl import cltypes

DIRNAME = path.abspath(path.dirname(__file__))

with open(path.join(DIRNAME, "hystopyramid.cl"), "r") as file:
    HYSTOPYRAMID_CL = file.read()


CELL_NBYTES = cltypes.float4.itemsize


def make_levels(depth, pyramid_g):
    origin = 0
    size = 1

    levels = []

    for _ in range(depth):
        sub = pyramid_g[CELL_NBYTES * origin : CELL_NBYTES * size * size]

        levels.append(sub)

        origin += size
        size *= 2

    return levels


def run(depth, bbox, points):
    grid_size = 2 ** (depth - 1)
    pyramid_ncells = int((4 ** depth - 1) / 3)
    pyramid_nbytes = CELL_NBYTES * pyramid_ncells

    points_np = np.array([cltypes.make_float3(*p) for p in points], cltypes.float3)
    pyramid_np = np.zeros(pyramid_ncells, cltypes.float4)

    ctx = cl.create_some_context()
    prg = cl.Program(ctx, HYSTOPYRAMID_CL).build(["-cl-std=CL2.0"])

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    mem = cl.mem_flags
    points_g = cl.Buffer(ctx, mem.READ_ONLY | mem.COPY_HOST_PTR, hostbuf=points_np)
    pyramid_g = cl.Buffer(ctx, mem.READ_WRITE, pyramid_nbytes)

    cl.enqueue_fill_buffer(queue, pyramid_g, cltypes.float(0), 0, pyramid_nbytes)

    levels = make_levels(depth, pyramid_g)

    ev_grid = prg.make_grid(
        queue,
        points_np.shape,
        None,
        cltypes.make_float4(*bbox),
        cltypes.uint(grid_size),
        points_g,
        levels[-1],
    )

    cl.enqueue_copy(queue, pyramid_np, pyramid_g)
    print(pyramid_np[: grid_size * grid_size], len(levels))

    events = [ev_grid]

    for lvl in range(len(levels) - 1, 0, -1):
        top = levels[lvl]
        cur = levels[lvl - 1]
        size = 2 ** (lvl - 1)
        ev_lvl = prg.make_level(queue, (size, size), None, cur.shape, top, cur)
        events.append(ev_lvl)

    cl.enqueue_copy(queue, pyramid_np, pyramid_g)

    spent = (events[-1].profile.end - ev_grid.profile.start) * 1e-6

    return pyramid_np, spent


def main():
    points = [(0.1, 0.1, 1), (0.9, 0.9, 1), (0.6, 0.6, 1)]

    (pyramid_np, spent) = run(1, (0, 0, 1, 1), points)

    print(pyramid_np, spent)


if __name__ == "__main__":
    main()
