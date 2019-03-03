from __future__ import absolute_import

import pyopencl as cl
import pyopencl.characterize.performance as perf


def main():
    ctx = cl.create_some_context()

    prof_overhead, latency = perf.get_profiling_overhead(ctx)
    print("Command latency: %.3f us" % (latency * 1e6))
    print(
        "Profiling overhead: %.3f us -> %.1f %%"
        % (prof_overhead * 1e6, 100 * prof_overhead / latency)
    )

    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    print("Empty kernel: %.3f us" % (perf.get_empty_kernel_time(queue) * 1e6))
    print("~~~")
    add_rate = perf.get_add_rate(queue)
    print("~~~")
    print("float32 add: %.3f GOps/s" % (add_rate / 1e9))


if __name__ == "__main__":
    main()
