import logging
import json
import os
import geopandas as gpd
import laspy
import numpy as np
import osmnx as ox
import time

from stl import mesh
from pathlib import Path
from shapely.geometry import shape
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime

from utils.logger import setup_logging
from tiles import (
    geodataframe_from_leaflet_to_ign, 
    geodataframe_from_ign_to_leaflet,
    init_folders,
    download_ign_available_tiles,
    merge_all_geojson_features,
    get_intersecting_tiles_from_order,
    download_tiles_from_gdf,
    decimate_array,
    filter_points_by_polygon,
    numpy_to_ply,
    meshlib_terrain_point_cloud_to_surface_mesh,
    add_base_to_surface_mesh
)


def execute_benchmark(order_folderpath:Path):
    FORCE_LAZ_TO_PLY = True
    Z_OFFSET = 10

    # Init folder tree if not existing
    init_folders()

    ox.utils.settings.cache_folder = Path('data/cache')

    # Download and merge all tiles availables
    FORCE_DOWNLOAD_ALL_TILES_AVAILABLE = False
    filepath_all_tiles_geojson         = Path('data/data_grille/all_tiles_available.geojson')
    filepath_all_tiles_geojson_merged  = Path('data/data_grille/all_tiles_available_merged.geojson')
    download_ign_available_tiles(filepath_all_tiles_geojson, FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)
    merge_all_geojson_features(filepath_all_tiles_geojson, filepath_all_tiles_geojson_merged, FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)

    laz_folderpath = Path('data/raw_point_cloud/')
    order_zone_filepath   = order_folderpath / 'order.json'
    order_intersects      = order_folderpath / 'tiles_intersect.geojson'
    ply_filepath          = order_folderpath / 'point_cloud.ply'
    surface_mesh_filepath = order_folderpath / 'surface_mesh.stl'
    final_mesh_filepath   = order_folderpath / 'final_mesh.stl'
    benchmark_result_filepath   = order_folderpath / f'benchmark_results--{int(100*datetime.now().timestamp())}.json'

    with open(benchmark_result_filepath, 'w') as f:
        f.write('{}')

    # Do the intersection if not already done
    if not os.path.isfile(order_intersects):
        gdf_intersect = get_intersecting_tiles_from_order(order_zone_filepath , filepath_all_tiles_geojson)
        gdf_intersect.to_file(order_intersects)
    
    # Download the non downloaded tiles
    gdf_intersect = gpd.read_file(order_intersects)
    download_tiles_from_gdf(gdf_intersect, laz_folderpath)
    

    choosen_point_classes = {
        'Ground': [2],
        'Ground + Buildings': [2, 6],
        'Ground + Vegetation + Buildings': [2, 3, 4, 5, 6, 9, 17],
        'All classes': [1, 2, 3, 4, 5, 6, 9, 17, 64, 66, 67]
    }
    percentage_point_to_remove_list = [99, 98, 97, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    smoothing_iteration_list = [0, 1, 2, 3]


    for percentage_point_to_remove in percentage_point_to_remove_list:
        for smoothing_iteration in smoothing_iteration_list:
            for point_class_title, choosen_point_class in choosen_point_classes.items():
                benchmark_id = f'{percentage_point_to_remove}--{smoothing_iteration}--{point_class_title.replace(' ', '')}'
                
                logger.info(f'\n\n STARTING NEW ITERATION\n benchmark_id : {benchmark_id}')

                data_benchmark = {
                    'smoothing': smoothing_iteration,
                    'percentage_point_to_remove': percentage_point_to_remove,
                    'point_class_title': point_class_title
                }


                # point cloud .laz to .ply + point decimation + point filtering by user zone selection
                start_time_filtering_decimating = time.time()
                polygon = gpd.read_file(order_zone_filepath).iloc[0].geometry
                logger.debug(f'polygon : {polygon}')
                list_intersecting_tiles_filename = list(gdf_intersect['name'])
                merged_xyz = list()
                total_all_original_points = 0
                for tile_filename in list_intersecting_tiles_filename:
                    tile_filepath = laz_folderpath / tile_filename

                    # Laz -> numpy   +   Remove unwanted points
                    las = laspy.read(tile_filepath)
                    choosen_xyz = np.empty((0,3))
                    total_all_original_points += len(las.x)
                    for point_class in choosen_point_class:
                        las_points = las.points[las.classification == point_class]
                        x = las_points.x.array
                        y = las_points.y.array
                        z = las_points.z.array
                        xyz = np.vstack([x,y,z]).T * las_points.scales   # Merged the coordinates into a (n,3) array and rescale them
                        choosen_xyz = np.vstack([choosen_xyz, xyz])
                    # Decimation of class-filtered points of the current tile
                    xyz = decimate_array(choosen_xyz, percentage_point_to_remove)
                    filtered_array = filter_points_by_polygon(xyz, polygon)
                    merged_xyz.append(filtered_array)                           # Merge the processed tile with the other tiles
                
                data_benchmark['total_all_original_points'] = total_all_original_points

                # Save all the needed points
                merged_xyz = np.vstack(merged_xyz)
                merged_xyz = merged_xyz - merged_xyz[0]     # Center the point cloud on (0,0,0)
                numpy_to_ply(merged_xyz, ply_filepath)

                duration_filtering_decimating = time.time() - start_time_filtering_decimating
                data_benchmark['duration_filtering_decimating'] = duration_filtering_decimating
                data_benchmark['total_all_choosen_points']  = len(merged_xyz)



                # Point cloud to surface mesh
                start_time_surface_to_mesh = time.time()
                meshlib_terrain_point_cloud_to_surface_mesh(ply_filepath, surface_mesh_filepath, smoothing=smoothing_iteration)
                duration_surface_to_mesh = time.time() - start_time_surface_to_mesh
                data_benchmark['duration_surface_to_mesh'] = duration_surface_to_mesh

                surface_mesh = mesh.Mesh.from_file(surface_mesh_filepath)
                data_benchmark['surface_mesh_triangles'] = len(surface_mesh.vectors)
                del surface_mesh

                # surface mesh to final mesh
                start_time_add_base_mesh = time.time()
                add_base_to_surface_mesh(surface_mesh_filepath, final_mesh_filepath, Z_OFFSET)
                duration_add_base_mesh = time.time() - start_time_add_base_mesh
                data_benchmark['duration_add_base_mesh'] = duration_add_base_mesh

                final_mesh = mesh.Mesh.from_file(final_mesh_filepath)
                data_benchmark['final_mesh_triangles'] = len(final_mesh.vectors)
                del final_mesh

                # Saving the benchmark data
                with open(benchmark_result_filepath, 'r') as f:
                    previous_data_benchmark = json.load(f)
                
                previous_data_benchmark[benchmark_id] = data_benchmark

                with open(benchmark_result_filepath, 'w') as f:
                    f.write(json.dumps(previous_data_benchmark, indent=4))



if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)
    execute_benchmark(Path('data/benchmark'))