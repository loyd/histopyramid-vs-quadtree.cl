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

TODO
