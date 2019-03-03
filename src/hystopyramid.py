from __future__ import absolute_import

import unittest
from os import path

import numpy as np
import pyopencl as cl
from numpy.testing import assert_allclose, assert_array_equal
from pyopencl import cltypes

DIRNAME = path.abspath(path.dirname(__file__))

with open(path.join(DIRNAME, "hystopyramid.cl"), "r") as file:
    HYSTOPYRAMID_CL = file.read()


CELL_NBYTES = cltypes.float4.itemsize


def float4(x, y, z, w):  # pylint: disable=invalid-name
    return cltypes.make_float4(x, y, z, w)


def make_levels(depth, pyramid_np, pyramid_g):
    origin = 0
    size = 1

    levels = []

    for _ in range(depth):
        end = origin + size * size
        level_np = pyramid_np[origin:end]
        level_g = pyramid_g[CELL_NBYTES * origin : CELL_NBYTES * end]

        levels.append((level_np, level_g))

        origin = end
        size *= 2

    return levels


def run(depth, bbox, points, ntimes=2, warmup=1):
    grid_size = 2 ** (depth - 1)
    pyramid_ncells = int((4 ** depth - 1) / 3)
    pyramid_nbytes = CELL_NBYTES * pyramid_ncells

    points_np = np.array([cltypes.make_float3(*p) for p in points], cltypes.float3)
    pyramid_np = np.zeros(pyramid_ncells, cltypes.float4)

    ctx = cl.create_some_context(False)
    prg = cl.Program(ctx, HYSTOPYRAMID_CL).build(["-cl-std=CL2.0"])

    make_grid_krn = prg.make_grid
    make_level_krn = prg.make_level

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    mem = cl.mem_flags
    points_g = cl.Buffer(ctx, mem.READ_ONLY | mem.COPY_HOST_PTR, hostbuf=points_np)
    pyramid_g = cl.Buffer(ctx, mem.READ_WRITE, pyramid_nbytes)

    levels = make_levels(depth, pyramid_np, pyramid_g)

    events = []

    for i in range(ntimes):
        cl.enqueue_fill_buffer(queue, pyramid_g, cltypes.float(0), 0, pyramid_nbytes)

        ev_grid = make_grid_krn(
            queue,
            points_np.shape,
            None,
            float4(*bbox),
            cltypes.uint(grid_size),
            points_g,
            levels[-1][1],
        )

        if i >= warmup:
            events.append(ev_grid)

        for lvl in range(len(levels) - 1, 0, -1):
            top_g = levels[lvl][1]
            cur_g = levels[lvl - 1][1]
            size = 2 ** (lvl - 1)
            ev_lvl = make_level_krn(queue, (size, size), None, top_g, cur_g)
            events.append(ev_lvl)

    cl.enqueue_copy(queue, pyramid_np, pyramid_g)
    queue.finish()

    return [lvl[0] for lvl in levels], events


class TestHystopyramid(unittest.TestCase):
    def test_single(self):
        zero = float4(0.0, 0.0, 0.0, 0.0)

        pyramid, _ = run(3, (0, 0, 1, 1), [(0.3, 0.7, 2.0)])
        self.assertEqual(len(pyramid), 3)
        assert_array_equal(pyramid[0], [float4(0.3, 0.7, 2.0, 1.0)])
        assert_array_equal(pyramid[1], [zero, zero, float4(0.3, 0.7, 2.0, 1.0), zero])

        pyramid, _ = run(3, (0, 0, 1, 1), [(0.3, 0.3, 2.0)])
        self.assertEqual(len(pyramid), 3)
        assert_array_equal(pyramid[0], [float4(0.3, 0.3, 2.0, 1.0)])
        assert_array_equal(pyramid[1], [float4(0.3, 0.3, 2.0, 1.0), zero, zero, zero])

        pyramid, _ = run(3, (0, 0, 1, 1), [(0.7, 0.7, 2.0)])
        self.assertEqual(len(pyramid), 3)
        assert_array_equal(pyramid[0], [float4(0.7, 0.7, 2.0, 1.0)])
        assert_array_equal(pyramid[1], [zero, zero, zero, float4(0.7, 0.7, 2.0, 1.0)])

        pyramid, _ = run(3, (0, 0, 1, 1), [(0.7, 0.3, 2.0)])
        self.assertEqual(len(pyramid), 3)
        assert_array_equal(pyramid[0], [float4(0.7, 0.3, 2.0, 1.0)])
        assert_array_equal(pyramid[1], [zero, float4(0.7, 0.3, 2.0, 1.0), zero, zero])

    def test_all_in(self):
        pyramid, _ = run(
            1, (0, 0, 1, 1), [(0.3, 0.2, 2.0), (0.7, 0.8, 1.0), (0.5, 0.5, 1.5)]
        )

        self.assertEqual(len(pyramid), 1)
        assert_array_equal(pyramid[0], [float4(1.5, 1.5, 4.5, 3.0)])

    def test_random(self):
        points = np.random.rand(100, 3)

        pyramid, _ = run(4, (0, 0, 1, 1), points)

        self.assertEqual(len(pyramid), 4)

        lhs = tuple(pyramid[0][0])
        rhs = (*points.sum(0), 100.0)

        self.assertTrue(np.allclose(lhs, rhs))


if __name__ == "__main__":
    unittest.main()
