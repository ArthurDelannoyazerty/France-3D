import wget
import logging
import os
import laspy
import requests
import json
import meshlib
import numpy as np
import pyvista as pv
import open3d as o3d
import geopandas as gpd

from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.vectorized import contains
from pathlib import Path
from utils.logger import setup_logging


logger = logging.getLogger(__name__)


crs_leaflet = 'EPSG:4326'
crs_ign =     'EPSG:2154'


def init_folders():
    logger.info('Create folders for the project')
    Path('data/data_grille'      ).mkdir(parents=True, exist_ok=True)
    Path('data/point_cloud/laz/' ).mkdir(parents=True, exist_ok=True)
    Path('data/point_cloud/ply/' ).mkdir(parents=True, exist_ok=True)
    Path('data/mesh'             ).mkdir(parents=True, exist_ok=True)
    Path('data/orders'           ).mkdir(parents=True, exist_ok=True)


def geodataframe_from_leaflet_to_ign(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.set_crs(crs_leaflet)
    gdf_transformed  = gdf.to_crs(crs_ign)
    return gdf_transformed

def geodataframe_from_ign_to_leaflet(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.set_crs(crs_ign)
    gdf_transformed  = gdf.to_crs(crs_leaflet)
    return gdf_transformed


def download_ign_available_tiles(output_filepath:str, force_download:bool=False):
    # If file already exists and no force, then do not download file
    if os.path.isfile(output_filepath) and not force_download: return

    wfs_url = "https://data.geopf.fr/private/wfs/?service=WFS&version=2.0.0&apikey=interface_catalogue&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:nuage-dalle&outputFormat=application/json"
    
    # First request to initialize the geojson and know the total number of features
    logger.info('First download for all features available')
    response = requests.get(wfs_url)
    if response.status_code != 200:
        logger.info(f"Failed to retrieve data for rul : {wfs_url}. HTTP Status code: {response.status_code}")
        exit(1)
    
    geojson = response.json()
    total_features = geojson['totalFeatures']
    number_returned = geojson['numberReturned']
    logger.info(f'First download finished. Total Feature : {total_features}  | Number Returned : {number_returned}')

    start_index = number_returned
    while start_index<total_features:
        logger.info(f'Downloading features from index {start_index} / {total_features}')
        wfs_url_indexed = f'{wfs_url}&startIndex={start_index}'
        response = requests.get(wfs_url_indexed)
        if response.status_code != 200:
            logger.info(f"Failed to retrieve data for rul : {wfs_url_indexed}. HTTP Status code: {response.status_code}")
            exit(1)
        response = response.json()
        current_features = response['features']
        geojson['features'].extend(current_features)
        number_returned = geojson['numberReturned']
        start_index += number_returned
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=4, ensure_ascii=False)


def merge_all_geojson_features(geojson_filepath:str, merged_geojson_filepath:str):
    logger.info(f'Merging all tiles from geojson : {geojson_filepath}')
    gdf = gpd.read_file(geojson_filepath)
    merged_gdf = unary_union(gdf.geometry)
    merged_gdf_dict = mapping(merged_gdf)
    with open(merged_geojson_filepath, 'w') as f:
        f.write(json.dumps(merged_gdf_dict))
    logger.info(f'Merged geojson saved at {merged_geojson_filepath}')


def numpy_to_laz(xyz:np.ndarray, old_laz_file:laspy.LasData, laz_filepath_out:str):
    logger.info(f'Saving point into laz at location {laz_filepath_out}')
    # Create new laz file with old laz header
    header = laspy.LasHeader(point_format=old_laz_file.header.point_format, version=old_laz_file.header.version)
    output_las = laspy.LasData(header)

    # Copy the points to the new file
    output_las.x = xyz[:, 0]
    output_las.y = xyz[:, 1]
    output_las.z = xyz[:, 2]
    output_las.write(laz_filepath_out)



def decimate_array(array:np.ndarray, percentage_to_remove):
    """
    Remove a given percentage of rows from a NumPy array randomly.

    Parameters:
    - array (np.ndarray): Input array of shape (n, 3).
    - percentage (float): Percentage of rows to remove (between 0 and 100).
    - seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - np.ndarray: Array with the specified percentage of rows removed.
    """
    logger.info(f'Decimating array of shape {array.shape} by {percentage_to_remove}%')
    if not 0 <= percentage_to_remove <= 100:
        raise ValueError("Percentage must be between 0 and 100.")
    
    mask = np.random.rand(array.shape[0]) > (percentage_to_remove / 100.0)
    decimated_array = array[mask]

    logger.info(f'Decimation done. Number of points : {array.shape[0]} (before) | {decimated_array.shape[0]} (after)')
    return decimated_array


def display_point_cloud(points:np.ndarray):
    """
    Display a 3D point cloud using PyVista.
    
    Parameters:
    - points (numpy.ndarray): A Nx3 array of 3D points (x, y, z).
    """
    logger.info('Display point cloud')
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_points(point_cloud, cmap="viridis", point_size=1)
    plotter.set_background("white")
    plotter.show()


def get_intersecting_tiles_from_order(geojson_order_filepath:str, geojson_all_tiles_available_filepath:str) -> gpd.GeoDataFrame:
    logger.info('Executing intersection of tiles for the order')
    logger.info('Loading geojson of all tiles available')
    available_tiles_gdf = gpd.read_file(geojson_all_tiles_available_filepath)
    
    logger.info('Loading geojson orders')
    order_gdf = gpd.read_file(geojson_order_filepath)
    logger.info(f'Order geojson head : {order_gdf.head()}')

    logger.info('Filtering the intersecting tiles')
    intersect_gdf = available_tiles_gdf[available_tiles_gdf.intersects(order_gdf.geometry.iloc[0])]
    logger.info(f'Intersect GeoDataFrame head : {intersect_gdf.head()}')
    return intersect_gdf


def download_tiles_from_gdf(gdf:gpd.GeoDataFrame, laz_folderpath:str):
    for index, row in gdf.iterrows():
        filename = row['name']
        url = row['url']
        filepath = laz_folderpath + filename
        if not os.path.isfile(filepath):
            logger.info(f'Downloading file {filename} into {filepath}')
            wget.download(url, out=filepath)


def filter_points_by_polygon(xyz, polygon):
    # Use shapely.vectorized.contains for efficient point-in-polygon testing
    logger.info(f'Filtering {xyz.shape[0]} points with the polygon {polygon}')
    inside_mask = contains(polygon, xyz[:, 0], xyz[:, 1])
    filtered_points = xyz[inside_mask]
    logger.info(f'Points filtered. {xyz.shape[0]} --> {filtered_points.shape[0]} ({filtered_points.shape[0]/xyz.shape[0]} %)')
    return filtered_points


def meshlib_terrain_point_cloud_to_mesh(ply_pointcloud_filepath:str, mesh_filepath:str):
    logger.info('Beginning Point cloud to mesh')
    pcd = o3d.io.read_point_cloud(ply_pointcloud_filepath)
    xyz = np.asarray(pcd.points) 
    xyz = xyz - xyz[0]
    vector_3d = meshlib.mrmeshnumpy.fromNumpyArray(xyz)

    logger.info('Triangulate terrain')
    mesh = meshlib.mrmeshpy.terrainTriangulation(vector_3d)

    logger.info('Remeshing')
    meshlib.mrmeshpy.remesh(mesh, meshlib.mrmeshpy.RemeshSettings())
    meshlib.mrmeshpy.saveMesh(mesh, mesh_filepath)



if __name__=="__main__":
    setup_logging()
    logger = logging.getLogger(__name__)


    FORCE_DOWNLOAD_ALL_TILES_AVAILABLE = False
    PERCENTAGE_POINT_TO_REMOVE = 0
    SHOW_CLOUDPOINT = True
    
    
    # Init folder tree if not existing
    init_folders()

    # Download and merge all tiles availables
    filepath_all_tiles_geojson        = 'data/data_grille/all_tiles_available.geojson'
    filepath_all_tiles_geojson_merged = 'data/data_grille/all_tiles_available_merged.geojson'
    if FORCE_DOWNLOAD_ALL_TILES_AVAILABLE : 
        download_ign_available_tiles(filepath_all_tiles_geojson)
        merge_all_geojson_features(filepath_all_tiles_geojson, filepath_all_tiles_geojson_merged)


    order_name = '1737817398--5XgZhio-_bDeS4AqAAAB'
    orders_folder  = 'data/orders/'
    laz_folderpath = 'data/point_cloud/laz/'
    order_filepath      = orders_folder + order_name + '/'
    order_zone_filepath = order_filepath + 'zone.geojson'
    order_intersects    = order_filepath + 'tiles_intersect.geojson'

    # Do the intersection if not already done
    if not os.path.isfile(order_intersects):
        gdf_intersect = get_intersecting_tiles_from_order(order_filepath , filepath_all_tiles_geojson)
        gdf_intersect.to_file(order_intersects)
    
    # Download the non downloaded tiles
    gdf_intersect = gpd.read_file(order_intersects)
    download_tiles_from_gdf(gdf_intersect, laz_folderpath)
    
    # point cloud .laz to .ply + point decimation + point filtering by user zone selection
    filtered_point_cloud_filepath = order_filepath + 'filtered_point_cloud.laz'
    if not os.path.isfile(filtered_point_cloud_filepath):
        polygon = gpd.read_file(order_zone_filepath).iloc[0].geometry
        list_intersecting_tiles_filename = list(gdf_intersect['name'])
        merged_xyz = list()
        for tile_filename in list_intersecting_tiles_filename:
            tile_filepath = laz_folderpath + tile_filename
            complete_laz_file = laspy.read(tile_filepath)
            xyz = complete_laz_file.xyz
            xyz = decimate_array(xyz, PERCENTAGE_POINT_TO_REMOVE)
            filtered_array = filter_points_by_polygon(xyz, polygon)
            if SHOW_CLOUDPOINT:
                display_point_cloud(xyz)
                display_point_cloud(filtered_array)
            merged_xyz.append(filtered_array)
        merged_xyz = np.array(merged_xyz).squeeze()
        numpy_to_laz(xyz=merged_xyz, old_laz_file=complete_laz_file, laz_filepath_out=filtered_point_cloud_filepath)


    if SHOW_CLOUDPOINT:
        filtered_laz = laspy.read(filtered_point_cloud_filepath)
        xyz = filtered_laz.xyz
        display_point_cloud(xyz)

    # Point cloud to mesh
    # mesh_filepath = order_filepath + 'mesh.stl'
    # meshlib_terrain_point_cloud_to_mesh(ply_filepath, mesh_filepath)
