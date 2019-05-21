# LIDAR-Processing

## Environment:
* `python`: 3.6.7
* `pdal`: 1.9.0
* `numpy`: 1.16.3
* `navpy`: 1.0
  
## How to run:
python create_fuse `[src_dir]` `[target_drc]` `[sample_rate]` `[num_processes]` `[save_to_fmt]` `[upper_left_lat]` `[upper_left_lon]` `[lower_right_lat]` `[lower_right_lon]` `[leve_of_detail]`

## Command-line args:
1. `src_dir`: source directory
2. `target_dir`: target directory to save the output fuses
3. `sample_rate`: decrease the sample rate by a factor of `sample_rate`
4. `num_processes`: number of processes to run simultaneously; cannot exceed `multiprocessing.cpu_count`.
5. `save_to_fmt`: format to save the fuses, e.g., .npy, .fuse
6, 7, 8, 9. `upper_left_lat`, `upper_left_lon`, `lower_right_lat`, `lower_right_lon': latitudes and longitudes of the bounding box.
10. `level_of_detail`: the same as `level of detail` specified in bing maps.


