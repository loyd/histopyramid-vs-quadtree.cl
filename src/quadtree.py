from __future__ import absolute_import

import unittest
from os import path

import numpy as np
import pyopencl as cl
from numpy.testing import assert_allclose, assert_array_equal
from pyopencl import cltypes

DIRNAME = path.abspath(path.dirname(__file__))

SHARED_DTYPE = np.dtype([("bbox", cltypes.float4), ("used", cltypes.int)])

NODE_DTYPE = np.dtype(
    [
        ("lock", cltypes.int),
        ("type", cltypes.int),
        ("count", cltypes.uint),
        ("_pad", cltypes.float),
        ("value", cltypes.float3),
        ("quarters", cltypes.int, 4),
    ]
)

with open(path.join(DIRNAME, "quadtree.cl"), "r") as file:
    QUADTREE_CL = file.read()


def run(max_depth, bbox, points, ntimes=2, warmup=1):
    quadtree_max_nodes = int((4 ** max_depth - 1) / 3)

    points_np = np.array([cltypes.make_float3(*p) for p in points], cltypes.float3)
    quadtree_np = np.zeros(quadtree_max_nodes, NODE_DTYPE)
    shared_np = np.array([(bbox, 0)], SHARED_DTYPE)

    ctx = cl.create_some_context(False)
    prg = cl.Program(ctx, QUADTREE_CL).build(
        ["-D", "MAX_DEPTH={}".format(max_depth), "-cl-std=CL2.0"]
    )

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    mem = cl.mem_flags
    points_g = cl.Buffer(ctx, mem.READ_ONLY | mem.COPY_HOST_PTR, hostbuf=points_np)
    quadtree_g = cl.Buffer(ctx, mem.READ_WRITE, quadtree_np.nbytes)
    shared_g = cl.Buffer(ctx, mem.READ_WRITE | mem.COPY_HOST_PTR, hostbuf=shared_np)

    events = []

    for i in range(ntimes):
        cl.enqueue_copy(queue, shared_g, shared_np)
        ev_run = prg.run(queue, points_np.shape, (1,), points_g, quadtree_g, shared_g)

        if i >= warmup:
            events.append(ev_run)

    cl.enqueue_copy(queue, quadtree_np, quadtree_g)
    cl.enqueue_copy(queue, shared_np, shared_g)
    queue.finish()

    return quadtree_np[: shared_np[0]["used"]], events


class TestQuadtree(unittest.TestCase):
    @staticmethod
    def get_values(quadtree):
        return list(tuple(float(f) for f in tuple(x)[:3]) for x in quadtree["value"])

    def assert_correctness(self, quadtree, node=None):
        node = node or quadtree[0]

        self.assertEqual(node["lock"], 0)
        self.assertNotEqual(node["type"], 0)

        quarters = node["quarters"]

        count = 0
        value = np.zeros(3)

        def consider(idx):
            nonlocal count, value

            node = quadtree[idx]
            self.assert_correctness(quadtree, node)
            count += node["count"]
            value += tuple(node["value"])[:3]

        if quarters[0] > 0:
            consider(quarters[0])
        if quarters[1] > 0:
            consider(quarters[1])
        if quarters[2] > 0:
            consider(quarters[2])
        if quarters[3] > 0:
            consider(quarters[3])

        if node["type"] == 2:
            self.assertEqual(node["count"], count)
            assert_allclose(tuple(node["value"])[:3], value, 1e-3)

    def test_single(self):
        quadtree, _ = run(2, (0, 0, 1, 1), [(0.3, 0.7, 2)])

        self.assertEqual(len(quadtree), 1)
        entry = quadtree[0]
        self.assert_correctness(quadtree)
        self.assertEqual(entry["type"], 1)
        self.assertEqual(entry["count"], 1)
        assert_allclose(TestQuadtree.get_values(quadtree), [(0.3, 0.7, 2.0)])
        self.assertTrue(not entry["quarters"].any())

    def test_opposite(self):
        quadtree, _ = run(4, (0, 0, 1, 1), [(0.3, 0.2, 2), (0.7, 0.8, 1)])

        self.assertEqual(len(quadtree), 3)
        self.assert_correctness(quadtree)
        assert_array_equal(quadtree["type"], (2, 1, 1))
        assert_array_equal(quadtree["count"], (2, 1, 1))
        assert_allclose(
            TestQuadtree.get_values(quadtree),
            [(1.0, 1.0, 3.0), (0.7, 0.8, 1.0), (0.3, 0.2, 2.0)],
        )

    def test_all_in(self):
        quadtree, _ = run(1, (0, 0, 1, 1), [(0.3, 0.2, 2), (0.7, 0.8, 1)])

        self.assertEqual(len(quadtree), 1)
        self.assert_correctness(quadtree)
        assert_array_equal(quadtree["type"], (1,))
        assert_array_equal(quadtree["count"], (2,))
        assert_allclose(TestQuadtree.get_values(quadtree), [(1.0, 1.0, 3.0)])

    def test_random(self):
        points = np.random.rand(100, 3)

        quadtree, _ = run(4, (0, 0, 1, 1), points)

        self.assertTrue(len(quadtree) > 0)
        top = quadtree[0]

        self.assertEqual(top["count"], 100)
        self.assert_correctness(quadtree)

        lhs = tuple(quadtree[0]["value"])[:3]
        rhs = tuple(points.sum(0))

        assert_allclose(lhs, rhs, 1e-3)


if __name__ == "__main__":
    unittest.main()
