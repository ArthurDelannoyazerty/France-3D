import wget
import logging
import os
import laspy
import requests
import json
import meshlib.mrmeshnumpy
import meshlib.mrmeshpy
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


def numpy_to_ply(xyz:np.ndarray, filepath_out:str):
    logger.info('Transforming numpy array into a .ply file')
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Save the point cloud to a .ply file
    o3d.io.write_point_cloud(filepath_out, pcd)


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


def display_point_cloud(points):
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
    logger.info(f'Points filtered. {xyz.shape[0]} --> {filtered_points.shape[0]} (Keeping {filtered_points.shape[0]/xyz.shape[0]} %)')
    return filtered_points


def meshlib_terrain_point_cloud_to_mesh(ply_pointcloud_filepath:str, mesh_filepath:str, smoothing:int=2):
    logger.info('Beginning Point cloud to mesh')
    pcd = o3d.io.read_point_cloud(ply_pointcloud_filepath)
    xyz = np.asarray(pcd.points) 
    xyz = xyz - xyz[0]
    vector_3d = meshlib.mrmeshnumpy.fromNumpyArray(xyz)

    logger.info('Triangulate terrain')
    mesh = meshlib.mrmeshpy.terrainTriangulation(vector_3d)

    logger.info('Remeshing')
    relax_params = meshlib.mrmeshpy.MeshRelaxParams()
    relax_params.iterations = smoothing  # Number of smoothing iterations

    # Apply the relaxation (smoothing) to the mesh
    meshlib.mrmeshpy.relax(mesh, relax_params)
    meshlib.mrmeshpy.saveMesh(mesh, mesh_filepath)



if __name__=="__main__":
    setup_logging()
    logger = logging.getLogger(__name__)


    FORCE_DOWNLOAD_ALL_TILES_AVAILABLE = False
    FORCE_LAZ_TO_PLY = True
    PERCENTAGE_POINT_TO_REMOVE = 50
    SHOW_CLOUDPOINT = False

    points_class_str_to_int = {
        'No Class':                      1,
        'Ground':                        2,
        'Small Vegetation (0-50cm)':     3,
        'Medium Vegetation (50-150 cm)': 4,
        'High Vegeration (+150 cm)':     5,
        'Building':                      6,
        'Water':                         9,
        'Bridge':                        17,
        'Perennial Soil':                64,
        'Virtual Points':                66,
        'Miscellaneous':                 67,
    }

    points_class_int_to_str = {
        1  : 'No Class',
        2  : 'Ground',
        3  : 'Small Vegetation (0-50cm)',
        4  : 'Medium Vegetation (50-150 cm)',
        5  : 'High Vegeration (+150 cm)',
        6  : 'Building',
        9  : 'Water',
        17 : 'Bridge',
        64 : 'Perennial Soil',
        66 : 'Virtual Points',
        67 : 'Miscellaneous',
    }

    choosen_point_class = [1, 2, 3, 4, 6, 17, 64]
    
    
    # Init folder tree if not existing
    init_folders()

    # Download and merge all tiles availables
    filepath_all_tiles_geojson        = 'data/data_grille/all_tiles_available.geojson'
    filepath_all_tiles_geojson_merged = 'data/data_grille/all_tiles_available_merged.geojson'
    if FORCE_DOWNLOAD_ALL_TILES_AVAILABLE : 
        download_ign_available_tiles(filepath_all_tiles_geojson)
        merge_all_geojson_features(filepath_all_tiles_geojson, filepath_all_tiles_geojson_merged)


    order_name = '1738280583--dA7XXUF3qlBSoJiZAAAB'
    orders_folder  = 'data/orders/'
    laz_folderpath = 'data/point_cloud/laz/'
    order_filepath      = orders_folder + order_name + '/'
    order_zone_filepath = order_filepath + 'zone.geojson'
    order_intersects    = order_filepath + 'tiles_intersect.geojson'

    # Do the intersection if not already done
    if not os.path.isfile(order_intersects):
        gdf_intersect = get_intersecting_tiles_from_order(order_zone_filepath , filepath_all_tiles_geojson)
        gdf_intersect.to_file(order_intersects)
    
    # Download the non downloaded tiles
    gdf_intersect = gpd.read_file(order_intersects)
    download_tiles_from_gdf(gdf_intersect, laz_folderpath)
    
    # point cloud .laz to .ply + point decimation + point filtering by user zone selection
    ply_filepath = order_filepath + 'point_cloud.ply'
    if not os.path.isfile(ply_filepath) or FORCE_LAZ_TO_PLY:
        polygon = gpd.read_file(order_zone_filepath).iloc[0].geometry
        list_intersecting_tiles_filename = list(gdf_intersect['name'])
        merged_xyz = list()
        for tile_filename in list_intersecting_tiles_filename:
            tile_filepath = laz_folderpath + tile_filename

            # Laz -> numpy   +   Remove unwanted points
            las = laspy.read(tile_filepath)
            choosen_xyz = np.empty((0,3))
            for point_class in choosen_point_class:
                las_points = las.points[las.classification == point_class]
                x = las_points.x.array
                y = las_points.y.array
                z = las_points.z.array
                xyz = np.vstack([x,y,z]).T * las_points.scales   # Merged the coordinates into a (n,3) array and rescale them
                choosen_xyz = np.vstack([choosen_xyz, xyz])

            # Decimation of class-filtered points of the current tile
            xyz = decimate_array(choosen_xyz, PERCENTAGE_POINT_TO_REMOVE)
            filtered_array = filter_points_by_polygon(xyz, polygon)
            merged_xyz.append(filtered_array)                           # Merge the processed tile with the other tiles

            if SHOW_CLOUDPOINT:
                display_point_cloud(xyz)
                display_point_cloud(filtered_array)
        # Save all the needed points
        merged_xyz = np.vstack(merged_xyz)
        merged_xyz = merged_xyz - merged_xyz[0]     # Center the point cloud on (0,0,0)
        numpy_to_ply(merged_xyz, ply_filepath)

    # Point cloud to mesh
    smoothing_iteration = 2
    mesh_filepath = order_filepath + 'mesh.stl'
    meshlib_terrain_point_cloud_to_mesh(ply_filepath, mesh_filepath, smoothing=smoothing_iteration)
