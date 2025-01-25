import sys
import wget
import logging
import os
import zipfile
import laspy
import numpy as np
import pyvista as pv
import laspy
import numpy as np
import open3d as o3d
import geopandas as gpd
import requests
import json
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union
import json

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


def laz_to_numpy(filepath_in) -> np.ndarray:
    logger.info('Transforming .laz into numpy array.')
    las = laspy.read(filepath_in)
    return las.xyz


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

    total_rows = array.shape[0]
    num_rows_to_remove = int(total_rows * (percentage_to_remove / 100.0))

    # Generate random indices to remove
    indices_to_remove = np.random.choice(total_rows, num_rows_to_remove, replace=False)

    # Create a mask that is True for rows to keep
    mask = np.ones(total_rows, dtype=bool)
    mask[indices_to_remove] = False

    # Return the array with the specified rows removed
    return array[mask]


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



def ply_pointcloud_to_ply_mesh_poisson(ply_pointcloud_filepath:str, 
                                       ply_mesh_filepath:str, 
                                       depth:int=8, 
                                       width:float=0, 
                                       scale:float=1.1, 
                                       linear_fit:bool=False):
    logger.info('Begin operation point cloud to mesh with poisson method.')
    
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_pointcloud_filepath)
    
    # Estimate normals
    pcd.estimate_normals()
    logger.info('Estimated normals done')

    # Point cloud to mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit
    )[0]
    logger.info('Poisson process done')

    logger.info('Estimating mesh vertex normals')
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

    # Save the mesh
    o3d.io.write_triangle_mesh(ply_mesh_filepath, mesh)


def xyz_to_mesh(ply_pointcloud_filepath:str, mesh_filepath:str):
    pcd = o3d.io.read_point_cloud(ply_pointcloud_filepath)
    xyz = np.asarray(pcd.points) 
    mesh:pv.pointset.PolyData = pv.wrap(xyz).delaunay_2d(alpha=0.0, progress_bar=True)
    mesh = mesh.compute_normals(progress_bar=True)
    mesh.save(mesh_filepath)
    return mesh



def display_ply_mesh(mesh_filepath:str):
    logger.info('Display mesh.')
    mesh = o3d.io.read_triangle_mesh(mesh_filepath)

    # Check if the mesh has vertex normals; if not, compute them
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


def download_tiles_order(geojson_order_filepath:str):
    pass





if __name__=="__main__":
    setup_logging()
    logger = logging.getLogger(__name__)


    FORCE_DOWNLOAD_ALL_TILES_AVAILABLE = False
    PERCENTAGE_POINT_TO_REMOVE = 25
    SHOW_CLOUDPOINT = False
    DO_POISSON = True
    
    
    # Init folder tree if not existing
    init_folders()

    filepath_all_tiles_geojson        = 'data/data_grille/all_tiles_available.geojson'
    filepath_all_tiles_geojson_merged = 'data/data_grille/all_tiles_available_merged.geojson'
    download_ign_available_tiles(filepath_all_tiles_geojson, force_download=FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)
    merge_all_geojson_features(filepath_all_tiles_geojson, filepath_all_tiles_geojson_merged)

    exit(0)
    # # Download zip if it doesn't exists
    # zip_output_dir = "data/data_grille/"
    # default_zip_name = 'grille.zip'
    # zip_filepath = zip_output_dir + default_zip_name
    # if not os.path.isfile(zip_filepath):
    #     download_lidar_ign_tiles_db(zip_output_dir)


    # # Extract files from zip
    # nb_files_in_data_grille_folder = len(os.listdir(zip_output_dir))
    # if nb_files_in_data_grille_folder==1:
    #     # If there is only one file in the folder, then the zip has not already been extracted
    #     extract_zip(zip_filepath, zip_output_dir)


    # # download tile
    # folder_data_cloudpoint_laz = 'data/point_cloud/laz/'
    # shp_file = zip_output_dir + 'TA_diff_pkk_lidarhd_classe.shp'

    # shp_df = gpd.read_file(shp_file, engine="pyogrio")
    # url_tile:str      = shp_df.iloc[INDEX_TILE_IGN].url_telech
    # filename_tile:str = shp_df.iloc[INDEX_TILE_IGN].nom_pkk
    # laz_filepath = folder_data_cloudpoint_laz + filename_tile
    
    # if not os.path.isfile(laz_filepath):
    #     logger.info(f'Downloading tile n°{INDEX_TILE_IGN} (.laz) into folder {folder_data_cloudpoint_laz}')
    #     laz_filepath = wget.download(url_tile, out=folder_data_cloudpoint_laz)
    #     logger.info(f'Tile downloaded into : {laz_filepath}')
    # else:
    #     logger.info(f'Tile n°{INDEX_TILE_IGN} is already downloaded at {laz_filepath}')


    # .laz to .ply
    folder_data_cloudpoint_ply = 'data/point_cloud/ply/'
    ply_filename = f'decimation-{str(PERCENTAGE_POINT_TO_REMOVE)}---{filename_tile.split('.')[0]}.ply'
    ply_filepath = folder_data_cloudpoint_ply + ply_filename

    if not os.path.isfile(ply_filepath):
        logger.info(f'Transformation .laz --> .ply  |  {laz_filepath} --> {ply_filepath}')
        xyz = laz_to_numpy(laz_filepath)
        xyz_decimated = decimate_array(xyz, PERCENTAGE_POINT_TO_REMOVE)
        logger.info(f'Decimation done. Number of points : {xyz.shape[0]} (before) | {xyz_decimated.shape[0]} (after)')
        numpy_to_ply(xyz_decimated, ply_filepath)
        logger.info(f'Transformation .laz --> .ply done')
        if SHOW_CLOUDPOINT: show_point_cloud(xyz_decimated)
    else:
        logger.info(f'File {ply_filepath} already exists.')

    
    # .ply point cloud to .stl mesh with delaunay 2D  (don't work with too much point)
    if DO_DELAUNAY:
        mesh_folder = 'data/mesh/'
        mesh_filename = f'delaunay---{ply_filename.split('.')[0]}.stl'
        mesh_filepath = mesh_folder + mesh_filename
        mesh = xyz_to_mesh(ply_filepath, mesh_filepath)

    # .ply point cloud to .ply mesh with ball pivoting
    if DO_BALL_PIVOTING:
        avg_radius = 4
        mesh_filename = f'ball_pivot--avg_radius-{avg_radius}---{ply_filename}'
        mesh_filepath = mesh_folder + mesh_filename
        ply_pointcloud_to_ply_mesh_ball_pivoting(ply_filepath, mesh_filepath, avg_radius)
        display_ply_mesh(mesh_filepath)

    
    # .ply point cloud to .ply mesh with poisson
    if DO_POISSON:
        depth = 10              # more = more detail & slower                                       -> Maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound.
        width = 0.0             # more = more/equal details & more outside anomalies                -> Specifies the target width of the finest level octree cells. This parameter is ignored if depth is specified
        scale = 1.0             # more = less details & more outside anomalies & faster             -> Specifies the ratio between the diameter of the cube used for reconstruction and the diameter of the samples’ bounding cube.
        linear_fit = True       #                                                                   -> If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.

        mesh_folder = 'data/mesh/'
        mesh_filename = f'poisson--depth-{int(depth*100)}--width-{int(width*100)}--scale-{int(scale*100)}--linear_fit-{linear_fit}---{ply_filename}'
        mesh_filepath = mesh_folder + mesh_filename

        ply_pointcloud_to_ply_mesh_poisson(ply_filepath, 
                                           mesh_filepath, 
                                           depth, 
                                           width, 
                                           scale, 
                                           linear_fit)
        display_ply_mesh(mesh_filepath)