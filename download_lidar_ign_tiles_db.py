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

from pathlib import Path
from utils.logger import setup_logging


logger = logging.getLogger(__name__)


def init_folders():
    logger.info('Create folders for the project')
    Path('data/data_grille'     ).mkdir(parents=True, exist_ok=True)
    Path('data/point_cloud/laz/').mkdir(parents=True, exist_ok=True)
    Path('data/point_cloud/ply/').mkdir(parents=True, exist_ok=True)
    Path('data/mesh'            ).mkdir(parents=True, exist_ok=True)

def download_ign_available_tiles():
    # Define the WFS URL
    wfs_url = (
        "https://data.geopf.fr/private/wfs/"
        "?service=WFS&version=2.0.0&apikey=interface_catalogue"
        "&request=GetFeature&typeNames=IGNF_LIDAR-HD_TA:nuage-dalle&outputFormat=application/json"
    )

    # Send a GET request to the WFS service
    response = requests.get(wfs_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON content
        data = response.json()

        # Define the output file path
        output_file = 'lidar_data.json'

        # Write the JSON data to a file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Data successfully downloaded and saved to {output_file}")
    else:
        print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")


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



if __name__=="__main__":
    setup_logging()
    logger = logging.getLogger(__name__)


    INDEX_TILE_IGN = 1000
    PERCENTAGE_POINT_TO_REMOVE = 25
    SHOW_CLOUDPOINT = False
    DO_DELAUNAY = False
    DO_BALL_PIVOTING = False
    DO_POISSON = True
    
    
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
    folder_data_cloudpoint_laz = 'data/point_cloud/laz/'
    shp_file = zip_output_dir + 'TA_diff_pkk_lidarhd_classe.shp'

    shp_df = gpd.read_file(shp_file, engine="pyogrio")
    url_tile:str      = shp_df.iloc[INDEX_TILE_IGN].url_telech
    filename_tile:str = shp_df.iloc[INDEX_TILE_IGN].nom_pkk
    laz_filepath = folder_data_cloudpoint_laz + filename_tile
    
    if not os.path.isfile(laz_filepath):
        logger.info(f'Downloading tile n°{INDEX_TILE_IGN} (.laz) into folder {folder_data_cloudpoint_laz}')
        laz_filepath = wget.download(url_tile, out=folder_data_cloudpoint_laz)
        logger.info(f'Tile downloaded into : {laz_filepath}')
    else:
        logger.info(f'Tile n°{INDEX_TILE_IGN} is already downloaded at {laz_filepath}')


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