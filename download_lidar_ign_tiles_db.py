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

from pathlib import Path
from utils.logger import setup_logging


logger = logging.getLogger(__name__)


def init_folders():
    logger.info('Create folders for the project')
    Path('data/data_grille'     ).mkdir(parents=True, exist_ok=True)
    Path('data/point_cloud/laz/').mkdir(parents=True, exist_ok=True)
    Path('data/point_cloud/ply/').mkdir(parents=True, exist_ok=True)
    Path('data/mesh'            ).mkdir(parents=True, exist_ok=True)


def download_lidar_ign_tiles_db(output_dir:str):
    url1 = "https://diffusion-lidarhd-classe.ign.fr/download/lidar/shp/classe"
    url2 = "https://zenodo.org/records/13793544/files/grille.zip"

    try:
        logger.info(f'Downloading IGN LIDAR Database from url : {url1}')
        wget.download(url=url1, out=output_dir)
    except:
        logger.info(f'Error while downloading from : {url1}  | Trying with : {url2}')
        try:
            wget.download(url=url2, out=output_dir)
        except:
            logger.error('Download failed for both url, shuting down...')
            sys.exit(1)
    logger.info(f'Files downloaded in : {output_dir}')


def extract_zip(filepath:str, folderpath:str):
    """Extract the zip file(filepath) to the desired folder(folderpath)."""
    logger.info(f'Extracting {filepath} into {folderpath}')
    with zipfile.ZipFile(filepath, 'r') as archive:
        archive.extractall(folderpath)


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


def show_point_cloud(points):
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


def ply_pointcloud_to_ply_mesh_ball_pivoting(ply_pointcloud_filepath:str, ply_mesh_filepath:str, avg_radius:float):
    logger.info('Begin operation point cloud to mesh with ball pivoting method.')
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(ply_pointcloud_filepath)

    # Estimate normals
    pcd.estimate_normals()
    logger.info('Estimated normals done')

    # Compute average distance between points
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_radius * avg_dist
    logger.info('Distances nearest neighbors done')

    # point cloud to mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius*2])
    )

    # Save the mesh
    o3d.io.write_triangle_mesh(ply_mesh_filepath, mesh)


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
        depth=8,
        width=0.0,
        scale=1.1,
        linear_fit=linear_fit
    )[0]

    # Save the mesh
    o3d.io.write_triangle_mesh(ply_mesh_filepath, mesh)



def display_ply_mesh(mesh_filepath:str):
    logger.info('Display mesh.')
    mesh = o3d.io.read_triangle_mesh(mesh_filepath)

    # Check if the mesh has vertex normals; if not, compute them
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])



if __name__=="__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Init folder tree if not existing
    init_folders()

    # Download zip if it doesn't exists
    zip_output_dir = "data/data_grille/"
    default_zip_name = 'grille.zip'
    zip_filepath = zip_output_dir + default_zip_name
    if not os.path.isfile(zip_filepath):
        download_lidar_ign_tiles_db(zip_output_dir)


    # Extract files from zip
    nb_files_in_data_grille_folder = len(os.listdir(zip_output_dir))
    if nb_files_in_data_grille_folder==1:
        # If there is only one file in the folder, then the zip has not already been extracted
        extract_zip(zip_filepath, zip_output_dir)


    # download tile
    index_tile = 1000       # Select a random tile for now
    folder_data_cloudpoint_laz = 'data/point_cloud/laz/'
    shp_file = zip_output_dir + 'TA_diff_pkk_lidarhd_classe.shp'

    shp_df = gpd.read_file(shp_file, engine="pyogrio")
    url_tile:str      = shp_df.iloc[index_tile].url_telech
    filename_tile:str = shp_df.iloc[index_tile].nom_pkk
    laz_filepath = folder_data_cloudpoint_laz + filename_tile
    
    if not os.path.isfile(laz_filepath):
        logger.info(f'Downloading tile n°{index_tile} (.laz) into folder {folder_data_cloudpoint_laz}')
        laz_filepath = wget.download(url_tile, out=folder_data_cloudpoint_laz)
        logger.info(f'Tile downloaded into : {laz_filepath}')
    else:
        logger.info(f'Tile n°{index_tile} is already downloaded at {laz_filepath}')


    # .laz to .ply
    SHOW_CLOUDPOINT = False
    percentage_point_to_remove = 95
    folder_data_cloudpoint_ply = 'data/point_cloud/ply/'
    ply_filename = f'decimation-{str(percentage_point_to_remove)}---{filename_tile.split('.')[0]}.ply'
    ply_filepath = folder_data_cloudpoint_ply + ply_filename

    if not os.path.isfile(ply_filepath):
        logger.info(f'Transformation .laz --> .ply  |  {laz_filepath} --> {ply_filepath}')
        xyz = laz_to_numpy(laz_filepath)
        xyz_decimated = decimate_array(xyz, percentage_point_to_remove)
        logger.info(f'Decimation done. Number of points : {xyz.shape[0]} (before) | {xyz_decimated.shape[0]} (after)')
        numpy_to_ply(xyz_decimated, ply_filepath)
        logger.info(f'Transformation .laz --> .ply done')
        if SHOW_CLOUDPOINT: show_point_cloud(xyz_decimated)
    else:
        logger.info(f'File {ply_filepath} already exists.')

    
    # .ply point cloud to .ply mesh with ball pivoting
    avg_radius = 4
    mesh_folder = 'data/mesh/'
    mesh_filename = f'ball_pivot-{ply_filepath}---avg_radius-{avg_radius}---{ply_filename}'
    mesh_filepath = mesh_folder + mesh_filename
    # ply_pointcloud_to_ply_mesh_ball_pivoting(ply_filepath, mesh_filepath, avg_radius)
    # display_ply_mesh(mesh_filepath)

    
    # .ply point cloud to .ply mesh with poisson
    depth = 8
    width = 0.0, 
    scale = 1.1, 
    linear_fit = False

    mesh_folder = 'data/mesh/'
    mesh_filename = f'poisson-{ply_filepath}---depth-{depth}---width-{width}---scale-{scale}---{ply_filename}'
    mesh_filepath = mesh_folder + mesh_filename

    ply_pointcloud_to_ply_mesh_poisson(ply_filepath, 
                                       mesh_filepath, 
                                       depth, 
                                       width, 
                                       scale, 
                                       linear_fit)
    display_ply_mesh(mesh_filepath)