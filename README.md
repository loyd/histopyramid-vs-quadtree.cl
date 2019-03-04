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

The results below are measured on GTX 1060.

Firstly, determine some characteristics of GPU using `src/checkperf.py`:
```
Command latency: 18.057 us
Profiling overhead: 4.037 us -> 22.4 %
Empty kernel: 25.643 us
float32 add: 52041.972 GOps/s
```

Then, run some benchmarks using `src/main.py`.

Legend:
* `H net`/`Q net` - spent time in **ms** on one iteration, GPU only.
* `H total`/`Q total` - spent time in **ms** on one iteration.

```
Depth   Points  H net   H total Q net   Q total
12      1       1.06    1.77    0.014   0.066
12      500     1.06    1.78    1.59    1.69
12      1000    1.06    1.78    3.22    3.32
12      2000    1.06    1.78    6.35    6.48
12      3000    1.07    1.79    7.42    7.57
12      4000    1.08    1.80    9.91    10.1
12      5000    1.06    1.78    12.4    12.5
12      6000    1.08    1.80    14.9    15.0
12      7000    1.09    1.82    17.4    17.5
12      8000    1.08    1.81    19.9    20.0
12      9000    1.11    1.84    22.3    22.5
12      10000   1.22    1.90    29.1    29.2
12      20000   1.23    1.99    49.1    49.3
12      30000   1.29    2.07    73.9    74.1
12      40000   1.40    2.21    98.7    98.9
12      50000   1.47    2.30    124     124
12      60000   1.57    2.44    150     150
12      70000   1.63    2.51    175     175
12      80000   1.74    2.65    199     200
12      90000   1.81    2.74    225     225
12      100000  1.91    2.87    250     250
12      200000  2.51    3.43
12      300000  3.59    5.03
12      400000  4.41    6.10
12      500000  5.24    7.15
12      600000  6.00    8.14
12      700000  6.78    9.15
12      800000  7.57    10.2
12      900000  8.35    11.2
12      1000000 9.09    12.1
```

```
Depth   Points  H net   H total
1       500000  1050    1050
2       500000  383     492
3       500000  79.6    103
4       500000  14.0    17.8
5       500000  1.82    2.33
6       500000  0.745   0.967
7       500000  0.602   0.782
8       500000  0.418   0.548
9       500000  0.987   1.29
10      500000  3.05    3.95
11      500000  4.17    5.47
12      500000  5.22    7.13
13      500000  8.34    12.4
14      500000  20.5    32.9
15      500000  69.2    115
```
