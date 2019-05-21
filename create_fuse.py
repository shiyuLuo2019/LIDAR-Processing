#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:22:53 2019

@author: luoshiyu
"""

import os
import shutil
import glob
import time
import sys
import multiprocessing
from multiprocessing import Pool

from TileSystem import TileSystem

import numpy as np
import pdal
import navpy
#from pykdtree.kdtree import KDTree


MIN_TILE_BUFFER = 0.001
MAX_TILE_BUFFER = 7.0
    
def get_drive_id(drive_fp):
    return drive_fp.split(os.sep)[-3]

def get_drive_dir_path(root_out_dir, drive_id):
    return os.path.join(root_out_dir, drive_id)

def get_drive_bool_dump_dir_path(drive_out_dir):
    return os.path.join(drive_out_dir, "masks")

def get_bool_array_fp(drive_out_dir, lat_or_lon, index):
    # example: lat_0.npy -> bool of first row in bbox
    return os.path.join(drive_out_dir, "%s_%d.npy" % (lat_or_lon, index))
    
def get_drive_tile_dir_path(drive_out_dir, quadkey):
    return os.path.join(drive_out_dir, quadkey)

def get_drive_tile_fuse_fp(drive_out_dir, quadkey, fmt):
    return os.path.join(get_drive_tile_dir_path(drive_out_dir, quadkey), '%s.%s' % (quadkey, fmt))
    
def get_drive_tile_meta_fp(drive_out_dir, quadkey):
    return os.path.join(get_drive_tile_dir_path(drive_out_dir, quadkey), '%s.meta' % quadkey)

def save_mask(fp, arr):
    compact = np.packbits(arr)
    np.save(fp, compact)
    
def load_mask(fp, n):
    compact = np.load(fp)
    return (np.unpackbits(compact).astype(np.bool))[:n]

def load_points(drive_fp, verbose=False):
    """
    Load the specified drive pointcloud.
    """
    pipeline="""{
          "pipeline": [
            {
                "type": "readers.las",
                "filename": "%s"
            }
          ]
        }""" % (drive_fp)
    p = pdal.Pipeline(pipeline)
    p.validate()
    p.execute()
    array = p.arrays[0]
    if verbose:
        print("%d points in %s" % (array.shape[0], drive_fp))
    lla = np.array((p.arrays[0]).tolist())[:, :3] # parse (lon, lat, alt)
    lla.T[[0, 1]] = lla.T[[1, 0]] #(lon, lat, alt) -> (lat, lon, alt)
    return lla

# =============================================================================
# def distances_greater_than_threshold(points_orthogonal, traj_orthogonal, threshold):
#     """
#     Determines for each row representing a point in a 3D orthogonal cordinate system
#     in <points_orthogonal> whether its distance to <traj_orthogonal> is greater than 
#     <threshold>.
#     """
#     traj_tree = KDTree(traj_orthogonal)
#     neighbor_dists, _ = traj_tree.query(points_orthogonal)
#     return neighbor_dists > threshold
# =============================================================================

def generate_edges(quadkeys, buf_size, use_buf):
    if use_buf: 
        assert(buf_size > MIN_TILE_BUFFER and buf_size < MAX_TILE_BUFFER)
        
    bboxes = [ ]
    for row in quadkeys:
        row_bboxes = [ ]
        for q in row:
            lat1, lon1, lat2, lon2 = TileSystem.quadKeyToBoundingBoxLatLong(q) # bboxes of original tile
            if use_buf:
                n1, e1 = buf_size, - buf_size
                ned2 = navpy.lla2ned(lat2, lon2, 0, lat_ref=lat1, lon_ref=lon1, alt_ref=0)
                n2, e2 = ned2[0] - buf_size, ned2[1] + buf_size
                lla1 = navpy.ned2lla(np.array([n1, e1, 0]), lat_ref=lat1, lon_ref=lon1, alt_ref=0)
                lla2 = navpy.ned2lla(np.array([n2, e2, 0]), lat_ref=lat1, lon_ref=lon1, alt_ref=0)
                row_bboxes.append([lla1[0], lla1[1],lla2[0], lla2[1]])
            else:
                row_bboxes.append([lat1, lon1, lat2, lon2])
        bboxes.append(row_bboxes)
    
    begin_edges_lat = [ ]
    end_edges_lat = [ ]
    for row_bboxes in bboxes:
        begin_edges_lat.append(row_bboxes[0][0])
        end_edges_lat.append(row_bboxes[0][2])
    
    begin_edges_lon = [ ]
    end_edges_lon = [ ]
    for bbox in bboxes[0]:
        begin_edges_lon.append(bbox[1])
        end_edges_lon.append(bbox[3])
    
    return bboxes, begin_edges_lat, end_edges_lat, begin_edges_lon, end_edges_lon


def run(drive_fp, drive_out_dir, lat1, lat2, lon1, lon2, sample_rate, level_of_detail=22, buf_size=0.2, use_buf=False, verbose=True, fmt='npy', remove_dump=False, drive_counter=0):

    dump_dir = get_drive_bool_dump_dir_path(drive_out_dir)
    if not os.path.exists(drive_out_dir):
        os.mkdir(drive_out_dir)
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
        
    drive_id = get_drive_id(drive_fp)
    if verbose: print("drive %s: loading points" % (drive_id))
    points = load_points(drive_fp)
    if verbose: print("drive %s: finished loading points, total %d points" % (drive_id, points.shape[0]))
    
    sample_rate = int(sample_rate)
    if verbose: print("drive %s: decreasing the sample rate by a factor of %d" % (drive_id, sample_rate))
    points = points[::sample_rate, :]
    if verbose: print("drive %s: %d points after downsampling" % (drive_id, points.shape[0]))
    
    # resize bbox
    min_lat, min_lon, _ = np.min(points, axis=0)
    min_lat = max(min_lat, lat2)
    min_lon = max(min_lon, lon1)
    max_lat, max_lon, _ = np.max(points, axis=0)
    max_lat = min(max_lat, lat1)
    max_lon = min(max_lon, lon2)
    if verbose: print("drive %s: resized bbox: (%f, %f), (%f, %f)" % (drive_id, max_lat, min_lon, min_lat, max_lon))
    quadkeys, _, _ = TileSystem.boundingBoxToTileQuadKeys2(max_lat, min_lon,min_lat, max_lon, level_of_detail)
    
    # generate edges
    
    _, begin_edges_lat, end_edges_lat, begin_edges_lon, end_edges_lon = generate_edges(quadkeys, buf_size=buf_size, use_buf=use_buf)
    if verbose: print("drive %s: finished generating tile edges. %d x %d tiles." % (drive_id, len(quadkeys), len(quadkeys[0])))
    
    if verbose: print("drive %s: histograming along lat" % drive_id)
    if use_buf:
        for r in range(len(begin_edges_lat)):
            save_mask(get_bool_array_fp(dump_dir, "lat", r), (points[:, 0] < begin_edges_lat[r]) & (points[:, 0] > end_edges_lat[r]))
    else:
        prev = None
        for r in range(len(begin_edges_lat) - 1, -1, -1):
            curr_bool = points[:, 0] < end_edges_lat[r]
            if prev is not None:
                save_mask(get_bool_array_fp(dump_dir, "lat", r + 1), curr_bool & prev)
            prev = ~curr_bool
        save_mask(get_bool_array_fp(dump_dir, "lat", 0), (points[:, 0] < begin_edges_lat[0]) & prev)
    if verbose: print("drive %s: finished histogramming along lat" % drive_id)
    
    if verbose: print("drive %s: histogramming along lon" % drive_id)
    if use_buf:
        for c in range(len(begin_edges_lon)):
            save_mask(get_bool_array_fp(dump_dir, "lon", c), (points[:, 1] > begin_edges_lon[c]) & (points[:, 1] < end_edges_lon[c]))
    else:
        prev = None
        for c in range(len(end_edges_lon) - 1, -1, -1):
            curr_bool = points[:, 1] > end_edges_lon[c]
            if prev is not None:
                save_mask(get_bool_array_fp(dump_dir, "lon", c + 1), curr_bool & prev)
            prev = ~curr_bool
        save_mask(get_bool_array_fp(dump_dir, "lon", 0), (points[:, 1] > begin_edges_lon[0]) & prev)
    if verbose: print("drive %s: finished histogramming along lon\n" % drive_id)
    
    used_points = 0
    num_tiles = len(quadkeys) * len(quadkeys[0])
    if verbose: print("drive %s: begin saving fuses, total %d tiles" % (drive_id, num_tiles))
    for r in range(len(quadkeys)):
        for c in range(len(quadkeys[r])):
            if verbose: print("drive %s: generating fuse for tile %s" % (drive_id, quadkeys[r][c]))
        
            # skip if no points in tile
            final_bool = load_mask(get_bool_array_fp(dump_dir, "lat", r), points.shape[0]) & load_mask(get_bool_array_fp(dump_dir, "lon", c), points.shape[0])
            curr_used_points = np.sum(final_bool)
            if curr_used_points == 0:
                continue
            
            # save fuse
            used_points += curr_used_points
            tile_dir = get_drive_tile_dir_path(drive_out_dir, quadkeys[r][c])
            if not os.path.exists(tile_dir):
                os.mkdir(tile_dir)
            if fmt == 'npy':
                np.save(get_drive_tile_fuse_fp(drive_out_dir, quadkeys[r][c], fmt), points[final_bool])
            else:
                np.savetxt(get_drive_tile_fuse_fp(drive_out_dir, quadkeys[r][c], fmt), points[final_bool], 
                           newline='\n', fmt='%.8f,%.8f,%.3f')
            if verbose: print("drive %s: finishied saving fuse for tile %s" % (drive_id, quadkeys[r][c]))
            del final_bool
            
            # meta
            with open(get_drive_tile_meta_fp(drive_out_dir, quadkeys[r][c]), 'w') as f:
                f.write('#drive-id:%s\n' % drive_id)
                f.write('#quadkey:%s\n' % quadkeys[r][c])
                f.write('#level-of-detail:%d\n' % level_of_detail)
                f.write('#global-bbox:(%f,%f),(%f,%f)\n' % (lat1, lon1, lat2, lon2))
                f.write('#used-bbox:(%f,%f),(%f,%f)\n' % (max_lat, min_lon, min_lat, max_lon))
                f.write('#buffer-on:{}\n'.format(use_buf))
                if use_buf: f.write('#buffer-size:%f\n' % (buf_size))
            
            num_tiles -= 1
            if verbose: print("drive %s: %d tiles remaining" % (drive_id, num_tiles))
            
            
    if verbose: print("drive %s: used points %d, total points %d" % (drive_id, used_points, points.shape[0]))
    assert(used_points == points.shape[0])
    
    if remove_dump:
        shutil.rmtree(dump_dir)
    
    print("finished drive #%d" % drive_counter)
            

if __name__ == "__main__":
    src_dir = sys.argv[1]
    target_dir = sys.argv[2]
    sample_rate = int(sys.argv[3])
    num_processes = int(sys.argv[4])
    fmt = sys.argv[5]
    lat1 = float(sys.argv[6])
    lon1 = float(sys.argv[7])
    lat2 = float(sys.argv[8])
    lon2 = float(sys.argv[9])
    level_of_detail = int(sys.argv[10])
    # other params
    buf_size = 0.15 # not used
    use_buf = False
    verbose = False
    remove_dump = True
    
    if not os.path.exists(src_dir):
        print("%s does not exist" % src_dir)
        sys.exit(0)
        
    if lat1 < lat2 or lon1 > lon2:
        print("illegal bounding box: upper-left: (%f, %f), lower-right: (%f, %f)" % (lat1, lon1, lat2, lon2))
        sys.exit(0)
    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print("created directory %s" % target_dir)
    
    num_processes = min(num_processes, multiprocessing.cpu_count())
    print("cpu count: %d, used: %d" % (multiprocessing.cpu_count(), num_processes))
    
    start_time = time.time()
    
    drives = glob.glob(os.path.join(src_dir, "*", "debug", "*", "*", "*.laz")) # PART A, B
    drives.extend(glob.glob(os.path.join(target_dir, "*", "*", "*", "*.laz"))) # PART C, D
    drives = drives[:4]
    print("found %d drives" % len(drives))
    
    p = Pool(processes=num_processes)
    
    drive_counter = 0
    for drive in drives:
        drive_out_dir = get_drive_dir_path(target_dir, get_drive_id(drive))
        p.apply_async(run,
                      args=[drive, drive_out_dir, lat1, lat2, lon1, lon2, 
                            sample_rate, level_of_detail, buf_size, use_buf, verbose, fmt, remove_dump, drive_counter]
                      )
        drive_counter += 1
       
    p.close()
    p.join()
    
    print("%d drives, time consumed: %d" % (len(drives), time.time() - start_time))
    
                    
            
        
        
        
    
        
    
    
    
    
    
    
    
    
    