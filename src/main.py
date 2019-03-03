from __future__ import absolute_import

import numpy as np

import hystopyramid
import quadtree


def fmt(nanos):
    millis = nanos / 1e6

    if millis < 1:
        tmpl = "%.3f"
    elif millis < 10:
        tmpl = "%.2f"
    elif millis < 100:
        tmpl = "%.1f"
    else:
        tmpl = "%.0f"

    return tmpl % millis


def inspect(events, verbose):
    rel = lambda x: fmt(x - events[0].profile.queued)
    spent_net = 0

    if verbose:
        print("~~~~~~~~~~~~   Kernels   ~~~~~~~~~~~~")
        print("Queued\tSubmit\tStart\tEnd\tDelta")

    for event in events:
        if verbose:
            print(
                "%f\t%s\t%s\t%s\t%s"
                % (
                    fmt(rel(event.profile.queued)),
                    fmt(rel(event.profile.submit)),
                    fmt(rel(event.profile.start)),
                    fmt(rel(event.profile.end)),
                    fmt((event.profile.end - event.profile.start)),
                )
            )

        spent_net += event.profile.end - event.profile.start

    spent_total = events[-1].profile.end - events[0].profile.start

    if verbose:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return spent_net, spent_total


def run_once(algo, depth, npoints, ntimes, warmup, verbose):
    points = np.random.rand(npoints, 3)

    _, events = algo.run(depth, (0, 0, 1, 1), points, ntimes, warmup)

    return inspect(events, verbose)


def run_through_npoints(depth, npoints_r, ntimes, warmup, include_quadtree):
    if include_quadtree:
        print("Depth\tPoints\tH net\tH total\tQ net\tQ total")
    else:
        print("Depth\tPoints\tH net\tH total")

    run_grid(range(depth, depth + 1), npoints_r, ntimes, warmup, include_quadtree)


def run_through_depth(npoints, depth_r, ntimes, warmup, include_quadtree):
    if include_quadtree:
        print("Depth\tPoints\tH net\tH total\tQ net\tQ total")
    else:
        print("Depth\tPoints\tH net\tH total")

    run_grid(depth_r, range(npoints, npoints + 1), ntimes, warmup, include_quadtree)


def run_grid(depth_r, npoints_r, ntimes, warmup, include_quadtree):
    for depth in depth_r:
        for npoints in npoints_r:
            hp_spent_net, hp_spent_total = run_once(
                hystopyramid, depth, npoints, ntimes, warmup, False
            )
            hp_spent_net = hp_spent_net / (ntimes - warmup)
            hp_spent_total = hp_spent_total / (ntimes - warmup)

            if include_quadtree:
                qt_spent_net, qt_spent_total = run_once(
                    quadtree, depth, npoints, ntimes, warmup, False
                )
                qt_spent_net = qt_spent_net / (ntimes - warmup)
                qt_spent_total = qt_spent_total / (ntimes - warmup)

            if include_quadtree:
                print(
                    "%d\t%d\t%s\t%s\t%s\t%s"
                    % (
                        depth,
                        npoints,
                        fmt(hp_spent_net),
                        fmt(hp_spent_total),
                        fmt(qt_spent_net),
                        fmt(qt_spent_total),
                    )
                )
            else:
                print(
                    "%d\t%d\t%s\t%s"
                    % (depth, npoints, fmt(hp_spent_net), fmt(hp_spent_total))
                )


def run_sample(algo, depth, npoints, warmup, ntimes):
    print(
        "Running %s: depth=%d points=%d iters=%d"
        % (algo.__name__, depth, npoints, ntimes)
    )

    spent_net, spent_total = run_once(algo, depth, npoints, ntimes, warmup, True)

    print("Spent net:   %s ms" % fmt(spent_net))
    print("Spent total: %s ms" % fmt(spent_total))


def main():
    run_through_npoints(8, range(10000, 150000, 25000), 10, 3, False)
    # run_through_depth(100000, range(3, 14), 10, 3, False)


if __name__ == "__main__":
    main()
