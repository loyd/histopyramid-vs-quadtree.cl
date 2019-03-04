# Quadtrees versus hystopyramids in OpenCL

## Usage

```sh
python3 src/main.py -h
```
```
usage: main.py [-h] [-q] [-p] -d DEPTH -n POINTS [-i ITERS] [-w WARMUP]

Comparison between quadtrees and hystopyramids
----------------------------------------------
Run and show details:
    main.py -q -h -d 6 -n 5000

Run through many depths:
    main.py -q -h -d 6:10:2 -n 5000

Run through many point counts:
    main.py -q -h -d 8 -n 1000:50000:10000

optional arguments:
  -h, --help            show this help message and exit
  -q, --quadtree
  -p, --hystopyramid
  -d DEPTH, --depth DEPTH
  -n POINTS, --points POINTS
  -i ITERS, --iters ITERS
  -w WARMUP, --warmup WARMUP
```

You might be interested in running `src/checkperf.py` before any benchmarks.

## Results

TODO: check using more performant GPU.

Legend:
* `H net`/`Q net` - spent time in **ms** on one iteration, GPU only.
* `H total`/`Q total` - spent time in **ms** on one iteration.

```
Depth   Points  H net   H total Q net   Q total
10      100     1.95    2.47    0.267   0.348
10      500     1.97    2.30    2.44    3.81
10      1000    1.89    3.27    5.36    5.60
10      2000    1.93    2.29    11.1    13.2
10      3000    2.11    2.50    16.2    16.5
10      4000    2.19    2.63    22.3    26.4
10      5000    2.17    2.65    28.5    29.7
10      6000    2.15    2.67    34.1    36.5
10      7000    2.19    2.70    39.8    43.2
10      8000    2.25    2.79    45.3    46.8
10      9000    2.33    2.88    50.8    52.7
10      10000   2.30    2.85    56.2    60.5
10      10000   2.45    2.99
10      20000   2.69    3.35
10      30000   3.19    4.06
10      40000   3.46    4.40
10      50000   3.91    5.05
10      60000   4.39    5.59
10      70000   4.86    6.20
10      80000   5.12    6.61
10      90000   5.80    7.28
10      100000  6.20    7.95
```

```
Depth   Points  H net   H total
1       200000  585     588
2       200000  73.2    94.5
3       200000  44.4    57.3
4       200000  14.8    19.2
5       200000  5.09    6.61
6       200000  4.32    5.76
7       200000  4.92    6.48
8       200000  6.57    8.71
9       200000  8.30    10.8
10      200000  11.5    14.5
11      200000  17.0    21.6
12      200000  34.4    51.8
13      200000  109     141
14      200000  400     645
```
