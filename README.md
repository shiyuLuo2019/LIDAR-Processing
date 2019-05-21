# LIDAR-Processing

## Environment:
* python: 3.6.7
* pdal: 1.9.0
* numpy: 1.16.3
* navpy: 1.0
  
## How to run:
python create_fuse `[src_dir]` `[target_drc]` `[sample_rate]` `[num_processes]` `[save_to_fmt]` `[upper_left_lat]` `[upper_left_lon]` `[lower_right_lat]` `[lower_right_lon]` `[leve_of_detail]`

### Command-line args
1. `src_dir`: source directory
2. tar
1. save_to_fmt: e.g., .npy, .fuse
sample_rate: must be integer; floats will be truncated into integer


