from __future__ import absolute_import

from os import path

import numpy as np
import pyopencl as cl
from pyopencl import cltypes

SHARED_DTYPE = np.dtype([("bbox", cltypes.float4), ("used", cltypes.int)])

NODE_DTYPE = np.dtype(
    [
        ("lock", cltypes.int),
        ("type", cltypes.int),
        ("count", cltypes.uint),
        ("_pad", cltypes.float),
        ("value", cltypes.float3),
        ("q0", cltypes.int),
        ("q1", cltypes.int),
        ("q2", cltypes.int),
        ("q3", cltypes.int),
    ]
)

with open(path.join(path.abspath(path.dirname(__file__)), "quadtree.cl"), "r") as file:
    QUADTREE_CL = file.read()


def run(max_depth, bbox, points):
    quadtree_max_size = int((4 ** max_depth - 1) / 3)

    points_np = np.array([(x, y, w, 0) for (x, y, w) in points], cltypes.float3)
    quadtree_np = np.zeros(quadtree_max_size, NODE_DTYPE)
    shared_np = np.array([(bbox, 0)], SHARED_DTYPE)

    ctx = cl.create_some_context()
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

    cl.enqueue_barrier(queue).wait()

    ev_sum = prg.run(queue, points_np.shape, (1,), points_g, quadtree_g, shared_g)

    cl.enqueue_copy(queue, quadtree_np, quadtree_g)
    cl.enqueue_copy(queue, shared_np, shared_g)

    spent = (ev_sum.profile.end - ev_sum.profile.start) * 1e-6

    return (quadtree_np[: shared_np[0]["used"]], spent)


def main():
    points = [(0.1, 0.1, 1), (0.9, 0.9, 1), (0.6, 0.6, 1)]

    (quadtree_np, spent) = run(5, (0, 0, 1, 1), points)

    print(quadtree_np, spent)


main()
