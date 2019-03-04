from __future__ import absolute_import

import argparse

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
    rel = lambda x: x - events[0].profile.queued
    spent_net = 0

    if verbose:
        print("~~~~~~~~~~~~   Kernels   ~~~~~~~~~~~~")
        print("Queued\tSubmit\tStart\tEnd\tDelta")

    for event in events:
        if verbose:
            print(
                "%s\t%s\t%s\t%s\t%s"
                % (
                    fmt(rel(event.profile.queued)),
                    fmt(rel(event.profile.submit)),
                    fmt(rel(event.profile.start)),
                    fmt(rel(event.profile.end)),
                    fmt(event.profile.end - event.profile.start),
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


def run_through_npoints(
    depth, npoints_r, ntimes, warmup, run_hystopyramid, run_quadtree
):
    header = "Depth\tPoints"

    if run_hystopyramid:
        header += "\tH net\tH total"

    if run_quadtree:
        header += "\tQ net\tQ total"

    print(header)

    run_grid(
        range(depth, depth + 1),
        npoints_r,
        ntimes,
        warmup,
        run_hystopyramid,
        run_quadtree,
    )


def run_through_depth(npoints, depth_r, ntimes, warmup, run_hystopyramid, run_quadtree):
    header = "Depth\tPoints"

    if run_hystopyramid:
        header += "\tH net\tH total"

    if run_quadtree:
        header += "\tQ net\tQ total"

    print(header)

    run_grid(
        depth_r,
        range(npoints, npoints + 1),
        ntimes,
        warmup,
        run_hystopyramid,
        run_quadtree,
    )


def run_grid(depth_r, npoints_r, ntimes, warmup, run_hystopyramid, run_quadtree):
    for depth in depth_r:
        for npoints in npoints_r:
            if run_hystopyramid:
                hp_spent_net, hp_spent_total = run_once(
                    hystopyramid, depth, npoints, ntimes, warmup, False
                )
                hp_spent_net = hp_spent_net / (ntimes - warmup)
                hp_spent_total = hp_spent_total / (ntimes - warmup)

            if run_quadtree:
                qt_spent_net, qt_spent_total = run_once(
                    quadtree, depth, npoints, ntimes, warmup, False
                )
                qt_spent_net = qt_spent_net / (ntimes - warmup)
                qt_spent_total = qt_spent_total / (ntimes - warmup)

            if run_hystopyramid and run_quadtree:
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
            elif run_hystopyramid:
                print(
                    "%d\t%d\t%s\t%s"
                    % (depth, npoints, fmt(hp_spent_net), fmt(hp_spent_total))
                )
            else:
                print(
                    "%d\t%d\t%s\t%s"
                    % (depth, npoints, fmt(qt_spent_net), fmt(qt_spent_total))
                )


def run_sample(depth, npoints, ntimes, warmup, run_hystopyramid, run_quadtree):
    if run_hystopyramid:
        print("## Hystopyramid")
        spent_net, spent_total = run_once(
            hystopyramid, depth, npoints, ntimes, warmup, True
        )
        print("H spent net:   %s ms" % fmt(spent_net))
        print("H spent total: %s ms" % fmt(spent_total))

    if run_hystopyramid and run_quadtree:
        print("\n")

    if run_quadtree:
        print("## Quadtree")
        spent_net, spent_total = run_once(
            quadtree, depth, npoints, ntimes, warmup, True
        )
        print("Q spent net:   %s ms" % fmt(spent_net))
        print("Q spent total: %s ms" % fmt(spent_total))


def parse_var(var):
    try:
        parts = [int(part) for part in var.split(":")]

        if len(parts) == 1:
            return parts[0]

        start = parts[0]
        end = parts[1]
        step = parts[2] if len(parts) > 2 else 100

        if start > end or step < 1:
            return None

        return range(start, end + 1, step)
    except:  # pylint: disable=bare-except
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare quadtree and hystopyramid")

    parser.add_argument("--quadtree", "-q", action="store_true")
    parser.add_argument("--hystopyramid", "-p", action="store_true")
    parser.add_argument("--depth", "-d", type=str, required=True)
    parser.add_argument("--points", "-n", type=str, required=True)
    parser.add_argument("--iters", "-i", type=int, default=10)
    parser.add_argument("--warmup", "-w", type=int, default=3)

    args = parser.parse_args()

    depth = parse_var(args.depth)
    points = parse_var(args.points)

    assert args.quadtree or args.hystopyramid, "You must add -q or/and -h"
    assert depth, "Depth must be scalar or valid range"
    assert points, "Point count must be scalar or valid range"
    assert not (
        isinstance(depth, range) and isinstance(points, range)
    ), "Only one of depth and point count can be range"
    assert args.iters > 0, "Iteration count must be positive"
    assert args.warmup > 0, "Warmup count must be positive"

    if isinstance(depth, range):
        run_through_depth(
            points, depth, args.iters, args.warmup, args.hystopyramid, args.quadtree
        )
    elif isinstance(points, range):
        run_through_npoints(
            depth, points, args.iters, args.warmup, args.hystopyramid, args.quadtree
        )
    else:
        run_sample(
            depth, points, args.iters, args.warmup, args.hystopyramid, args.quadtree
        )


if __name__ == "__main__":
    main()
